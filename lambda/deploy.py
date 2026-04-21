#!/usr/bin/env python3
"""
Automated deployment script for the OCR + Bedrock Lambda pipeline.

Creates a SEPARATE S3 bucket with all models (PaddleOCR + YOLO masking),
ECR + OCR Lambda (container), Aadhaar masking Lambda (zip + layer), IAM, DynamoDB, S3 triggers.

Place masking deployment zip at: chola-mandapam/masking_lambda/deployment.zip
(export from source account; may be >50 MB — uploaded to S3 then deployed.)

Publish layer in target account first, e.g.:
  aws lambda publish-layer-version --layer-name equitas-masking-onnx --zip-file fileb://layer.zip ...

If the masking handler imports numpy (etc.) and you see ImportError, either bundle those
deps inside deployment.zip or publish extra layers and set MASKING_EXTRA_LAYER_NAMES
(e.g. numpy from masking_lambda/build_numpy_layer.sh). Unzipped code + layers must be
≤ 250 MB; with a numpy layer, enable MASKING_STRIP_BUNDLED_NUMPY_FROM_ZIP to drop
duplicate numpy trees from deployment.zip.

Usage:
  python deploy.py --profile my-sso-profile
  python deploy.py --profile my-sso-profile --skip-models
  python deploy.py --profile my-sso-profile --skip-masking
  python deploy.py --profile my-sso-profile --update-only
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import logging
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# ── Configuration ────────────────────────────────────────────────────────────

REGION = "ap-south-1"
BEDROCK_REGION = "ap-south-1"

# ACCOUNT_ID and S3_BUCKET are resolved at runtime via STS — see _resolve_account_id()
ACCOUNT_ID: str = ""
S3_BUCKET: str = ""


def _resolve_account_id(session: boto3.Session) -> None:
    """Auto-detect AWS account ID from caller identity and derive bucket name."""
    global ACCOUNT_ID, S3_BUCKET
    if ACCOUNT_ID:
        return
    sts = session.client("sts")
    ACCOUNT_ID = sts.get_caller_identity()["Account"]
    S3_BUCKET = f"chola-ocr-pipeline-{ACCOUNT_ID}"
    log.info("Resolved account: %s, bucket: %s", ACCOUNT_ID, S3_BUCKET)
MODEL_PREFIX = "models/paddleocr"
YOLO_MODEL_S3_KEY = "model/best_new.onnx"

# ECR
ECR_REPO = "chola-ocr-pipeline"
IMAGE_TAG = "latest"

# Lambda — OCR pipeline
LAMBDA_NAME = "chola-ocr-pipeline"
LAMBDA_MEMORY = 6144
LAMBDA_TIMEOUT = 300
LAMBDA_STORAGE = 1024
LAMBDA_ARCH = "x86_64"

# Lambda — Aadhaar masking (zip + ONNX layer; not container)
MASKING_LAMBDA_NAME = "aadhar-masking"
MASKING_ROLE_NAME = "aadhar-masking-lambda-role"
MASKING_LAYER_NAME = "equitas-masking-onnx"
# Optional layers (latest version each), applied before the ONNX layer.
# Publish masking_lambda/numpy-layer.zip, then set the same --layer-name here.
MASKING_EXTRA_LAYER_NAMES: tuple[str, ...] = ("aadhar-masking-numpy",)
# If deployment.zip still contains numpy (from the source export), strip it before upload
# when a numpy layer is attached — otherwise code+layers can exceed 250 MB unzipped.
MASKING_STRIP_BUNDLED_NUMPY_FROM_ZIP = True
MASKING_LAMBDA_HANDLER = "handler.lambda_handler"
MASKING_LAMBDA_RUNTIME = "python3.11"
MASKING_LAMBDA_MEMORY = 6144
MASKING_LAMBDA_TIMEOUT = 900  # 15 minutes
MASKING_LAMBDA_STORAGE = 1024
# Deployment package (export from source env). Parent = chola-mandapam/
MASKING_DEPLOY_ZIP = Path(__file__).parent.parent / "masking_lambda" / "deployment.zip"
# Staging key for CreateFunction (supports packages > direct 50 MB limit)
MASKING_S3_DEPLOY_KEY = "lambda-deploy/masking/deployment.zip"

# IAM — OCR pipeline
ROLE_NAME = "chola-ocr-pipeline-lambda-role"

# HuggingFace model details
HF_REPO_ID = "monkt/paddleocr-onnx"
HF_MODEL_FILES = {
    "det.onnx": "detection/v5/det.onnx",
    "rec.onnx": "languages/english/rec.onnx",
    "dict.txt": "languages/english/dict.txt",
}

# Local YOLO model path (relative to project root)
YOLO_MODEL_LOCAL = Path(__file__).parent.parent / "yolo_masking_model" / "best_new.onnx"

DYNAMO_TABLE = "chola-ocr-pipeline-logs"

# Resource tags for the OCR Lambda (org / compliance — align with org tag policy)
LAMBDA_RESOURCE_TAGS: dict[str, str] = {
    "APP-NAME": "CA-DEVSECOPS",
    "BU": "GB",
    "ENVIRONMENT": "NON-PROD",
    "map-migrated": "migPQG47ENTCM",
    "Name": "ca-devsecops",
}

# Masking Lambda tags (same org keys; Name identifies the function)
MASKING_LAMBDA_TAGS: dict[str, str] = {
    **LAMBDA_RESOURCE_TAGS,
    "Name": "aadhar-masking",
}


def _build_masking_lambda_env() -> dict:
    """Env vars for masking Lambda (extend if handler expects more keys)."""
    return {
        "S3_BUCKET": S3_BUCKET,
    }


def _lambda_function_arn() -> str:
    """Full ARN for the pipeline Lambda (requires _resolve_account_id() first)."""
    return f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:{LAMBDA_NAME}"


def _masking_function_arn() -> str:
    """Full ARN for the masking Lambda (requires _resolve_account_id() first)."""
    return f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:{MASKING_LAMBDA_NAME}"


def _apply_lambda_tags(lam) -> None:
    """Apply standard resource tags (create may already set them; update path needs this)."""
    try:
        lam.tag_resource(Resource=_lambda_function_arn(), Tags=LAMBDA_RESOURCE_TAGS)
        log.info("  Lambda resource tags applied.")
    except ClientError as exc:
        log.warning("  Lambda tag_resource skipped: %s", exc)


def _apply_masking_lambda_tags(lam) -> None:
    try:
        lam.tag_resource(Resource=_masking_function_arn(), Tags=MASKING_LAMBDA_TAGS)
        log.info("  Masking Lambda resource tags applied.")
    except ClientError as exc:
        log.warning("  Masking Lambda tag_resource skipped: %s", exc)


def _build_lambda_env_vars() -> dict:
    """Build Lambda env vars dict (must be called after _resolve_account_id)."""
    return {
        "S3_BUCKET": S3_BUCKET,
        "OUTPUT_PREFIX": "pipeline_output",
        "MASKING_PREFIX": "chola_input",
        "BEDROCK_MODEL": "mistral.ministral-3-14b-instruct",
        "BEDROCK_REGION": BEDROCK_REGION,
        "MODEL_BUCKET": S3_BUCKET,
        "MODEL_PREFIX": MODEL_PREFIX,
        "DYNAMO_TABLE": DYNAMO_TABLE,
    }

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("deploy")

SCRIPT_DIR = Path(__file__).parent


# ── Step 1: Create S3 bucket ────────────────────────────────────────────────

def ensure_s3_bucket(session):
    """Create the dedicated S3 bucket if it doesn't exist."""
    log.info("=== Step 1: Ensure S3 bucket ===")

    s3 = session.client("s3", region_name=REGION)

    try:
        s3.head_bucket(Bucket=S3_BUCKET)
        log.info("  Bucket exists: %s", S3_BUCKET)
        return
    except ClientError:
        pass

    log.info("  Creating bucket: %s (region: %s)", S3_BUCKET, REGION)
    s3.create_bucket(
        Bucket=S3_BUCKET,
        CreateBucketConfiguration={"LocationConstraint": REGION},
    )

    # Block public access
    s3.put_public_access_block(
        Bucket=S3_BUCKET,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    log.info("  Bucket created with public access blocked.")


# ── Step 2: Upload models to S3 ─────────────────────────────────────────────

def upload_models(session):
    """Upload PaddleOCR models (from HuggingFace) and YOLO model to S3."""
    log.info("=== Step 2: Upload models to S3 ===")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        log.error(
            "huggingface_hub not installed. Run: pip install huggingface_hub"
        )
        sys.exit(1)

    s3 = session.client("s3", region_name=REGION)

    # PaddleOCR models
    for filename, hf_path in HF_MODEL_FILES.items():
        s3_key = f"{MODEL_PREFIX}/{filename}"

        try:
            s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
            log.info("  Already exists: s3://%s/%s", S3_BUCKET, s3_key)
            continue
        except ClientError:
            pass

        log.info("  Downloading %s from HuggingFace …", hf_path)
        local_path = hf_hub_download(HF_REPO_ID, hf_path)

        log.info("  Uploading → s3://%s/%s", S3_BUCKET, s3_key)
        s3.upload_file(local_path, S3_BUCKET, s3_key)

    # YOLO masking model
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=YOLO_MODEL_S3_KEY)
        log.info("  Already exists: s3://%s/%s", S3_BUCKET, YOLO_MODEL_S3_KEY)
    except ClientError:
        if not YOLO_MODEL_LOCAL.exists():
            log.error("  YOLO model not found at %s", YOLO_MODEL_LOCAL)
            sys.exit(1)
        log.info(
            "  Uploading YOLO model (%d MB) → s3://%s/%s",
            YOLO_MODEL_LOCAL.stat().st_size // (1024 * 1024),
            S3_BUCKET,
            YOLO_MODEL_S3_KEY,
        )
        s3.upload_file(str(YOLO_MODEL_LOCAL), S3_BUCKET, YOLO_MODEL_S3_KEY)

    log.info("  All models ready in S3.")


# ── Step 3: Create ECR repository ───────────────────────────────────────────

def ensure_ecr_repo(session) -> str:
    """Create ECR repo if it doesn't exist. Return the repository URI."""
    log.info("=== Step 3: Ensure ECR repository ===")

    ecr = session.client("ecr", region_name=REGION)

    try:
        resp = ecr.describe_repositories(repositoryNames=[ECR_REPO])
        uri = resp["repositories"][0]["repositoryUri"]
        log.info("  ECR repo exists: %s", uri)
        return uri
    except ClientError as e:
        if e.response["Error"]["Code"] != "RepositoryNotFoundException":
            raise

    resp = ecr.create_repository(
        repositoryName=ECR_REPO,
        imageScanningConfiguration={"scanOnPush": True},
    )
    uri = resp["repository"]["repositoryUri"]
    log.info("  Created ECR repo: %s", uri)
    return uri


# ── Step 4: Build & push Docker image ───────────────────────────────────────

def build_and_push_image(session, ecr_uri: str):
    """Build the Docker image and push to ECR."""
    log.info("=== Step 4: Build & push Docker image ===")

    image_full = f"{ecr_uri}:{IMAGE_TAG}"

    # Authenticate Docker to ECR
    ecr = session.client("ecr", region_name=REGION)
    token_resp = ecr.get_authorization_token()
    auth_data = token_resp["authorizationData"][0]
    token = base64.b64decode(auth_data["authorizationToken"]).decode()
    username, password = token.split(":", 1)
    registry = auth_data["proxyEndpoint"]

    log.info("  Authenticating Docker to ECR …")
    _run_cmd(
        ["docker", "login", "--username", username, "--password-stdin", registry],
        input_data=password,
    )

    # Build image for x86_64 (Lambda architecture)
    log.info("  Building Docker image (platform linux/amd64) …")
    _run_cmd([
        "docker", "buildx", "build",
        "--platform", "linux/amd64",
        "--provenance=false",
        "-t", image_full,
        "--load",
        str(SCRIPT_DIR),
    ])

    # Push to ECR
    log.info("  Pushing image to ECR …")
    _run_cmd(["docker", "push", image_full])

    log.info("  Image pushed: %s", image_full)
    return image_full


# ── Step 5: Create IAM role ─────────────────────────────────────────────────

def ensure_iam_role(session) -> str:
    """Create IAM role + policy if they don't exist. Return role ARN."""
    log.info("=== Step 5: Ensure IAM role ===")

    iam = session.client("iam")

    # Check if role exists
    try:
        resp = iam.get_role(RoleName=ROLE_NAME)
        role_arn = resp["Role"]["Arn"]
        log.info("  IAM role exists: %s", role_arn)

        # Update policy in case bucket name changed
        policy_doc = _build_iam_policy()
        iam.put_role_policy(
            RoleName=ROLE_NAME,
            PolicyName="chola-ocr-pipeline-policy",
            PolicyDocument=json.dumps(policy_doc),
        )
        log.info("  Updated inline policy.")
        return role_arn
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            raise

    # Create role with Lambda trust policy
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    resp = iam.create_role(
        RoleName=ROLE_NAME,
        AssumeRolePolicyDocument=json.dumps(trust_policy),
        Description="Execution role for chola-ocr-pipeline Lambda",
    )
    role_arn = resp["Role"]["Arn"]
    log.info("  Created IAM role: %s", role_arn)

    # Attach inline policy
    policy_doc = _build_iam_policy()
    iam.put_role_policy(
        RoleName=ROLE_NAME,
        PolicyName="chola-ocr-pipeline-policy",
        PolicyDocument=json.dumps(policy_doc),
    )
    log.info("  Attached inline policy.")

    # Wait for role to propagate (IAM is eventually consistent)
    log.info("  Waiting 10s for IAM role propagation …")
    time.sleep(10)

    return role_arn


def ensure_masking_iam_role(session) -> str:
    """Create IAM role + inline policy for the Aadhaar masking Lambda."""
    log.info("=== Step 5b: Ensure masking Lambda IAM role ===")

    iam = session.client("iam")
    policy_doc = _build_masking_iam_policy()

    try:
        resp = iam.get_role(RoleName=MASKING_ROLE_NAME)
        role_arn = resp["Role"]["Arn"]
        log.info("  Masking IAM role exists: %s", role_arn)
        iam.put_role_policy(
            RoleName=MASKING_ROLE_NAME,
            PolicyName="aadhar-masking-lambda-policy",
            PolicyDocument=json.dumps(policy_doc),
        )
        log.info("  Updated masking inline policy.")
        return role_arn
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            raise

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    resp = iam.create_role(
        RoleName=MASKING_ROLE_NAME,
        AssumeRolePolicyDocument=json.dumps(trust_policy),
        Description="Execution role for Aadhaar masking Lambda (S3 read/write)",
    )
    role_arn = resp["Role"]["Arn"]
    log.info("  Created masking IAM role: %s", role_arn)

    iam.put_role_policy(
        RoleName=MASKING_ROLE_NAME,
        PolicyName="aadhar-masking-lambda-policy",
        PolicyDocument=json.dumps(policy_doc),
    )
    log.info("  Attached masking inline policy.")

    log.info("  Waiting 10s for IAM role propagation …")
    time.sleep(10)
    return role_arn


def _build_masking_iam_policy() -> dict:
    """S3 access for masking: inputs, YOLO model, outputs."""
    b = S3_BUCKET
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "S3ReadMasking",
                "Effect": "Allow",
                "Action": "s3:GetObject",
                "Resource": [
                    f"arn:aws:s3:::{b}/chola_input/*",
                    f"arn:aws:s3:::{b}/model/*",
                    f"arn:aws:s3:::{b}/chola_result/*",
                ],
            },
            {
                "Sid": "S3WriteMasking",
                "Effect": "Allow",
                "Action": "s3:PutObject",
                "Resource": [
                    f"arn:aws:s3:::{b}/chola_input/*",
                    f"arn:aws:s3:::{b}/chola_result/*",
                ],
            },
            {
                "Sid": "CloudWatchLogsMasking",
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                "Resource": f"arn:aws:logs:{REGION}:{ACCOUNT_ID}:*",
            },
        ],
    }


def _build_iam_policy() -> dict:
    """Build IAM policy document for the current bucket name."""
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "S3Read",
                "Effect": "Allow",
                "Action": "s3:GetObject",
                "Resource": f"arn:aws:s3:::{S3_BUCKET}/*",
            },
            {
                "Sid": "S3Write",
                "Effect": "Allow",
                "Action": "s3:PutObject",
                "Resource": [
                    f"arn:aws:s3:::{S3_BUCKET}/pipeline_output/*",
                    f"arn:aws:s3:::{S3_BUCKET}/chola_input/*",
                    f"arn:aws:s3:::{S3_BUCKET}/pipeline_pages/*",
                ],
            },
            {
                "Sid": "BedrockInvoke",
                "Effect": "Allow",
                "Action": ["bedrock:InvokeModel"],
                "Resource": "arn:aws:bedrock:ap-south-1::foundation-model/mistral.*",
            },
            {
                "Sid": "DynamoDBLogs",
                "Effect": "Allow",
                "Action": [
                    "dynamodb:PutItem",
                    "dynamodb:UpdateItem",
                    "dynamodb:GetItem",
                ],
                "Resource": f"arn:aws:dynamodb:{REGION}:{ACCOUNT_ID}:table/{DYNAMO_TABLE}",
            },
            {
                "Sid": "CloudWatchLogs",
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                "Resource": f"arn:aws:logs:{REGION}:{ACCOUNT_ID}:*",
            },
        ],
    }


# ── Step 6: Create or update Lambda function ────────────────────────────────

def ensure_lambda(session, image_uri: str, role_arn: str):
    """Create Lambda function or update its image if it already exists."""
    log.info("=== Step 6: Ensure Lambda function ===")

    lam = session.client("lambda", region_name=REGION)

    try:
        lam.get_function(FunctionName=LAMBDA_NAME)
        log.info("  Lambda exists, updating image + env vars …")

        lam.update_function_code(
            FunctionName=LAMBDA_NAME,
            ImageUri=image_uri,
        )
        _wait_for_lambda_update(lam, LAMBDA_NAME)

        lam.update_function_configuration(
            FunctionName=LAMBDA_NAME,
            Environment={"Variables": _build_lambda_env_vars()},
        )
        _wait_for_lambda_update(lam, LAMBDA_NAME)

        _apply_lambda_tags(lam)

        log.info("  Lambda updated.")
        return

    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise

    # Create new Lambda
    log.info("  Creating Lambda function: %s", LAMBDA_NAME)
    lam.create_function(
        FunctionName=LAMBDA_NAME,
        PackageType="Image",
        Code={"ImageUri": image_uri},
        Role=role_arn,
        Timeout=LAMBDA_TIMEOUT,
        MemorySize=LAMBDA_MEMORY,
        EphemeralStorage={"Size": LAMBDA_STORAGE},
        Architectures=[LAMBDA_ARCH],
        Environment={"Variables": _build_lambda_env_vars()},
        Tags=LAMBDA_RESOURCE_TAGS,
    )

    _wait_for_lambda_update(lam, LAMBDA_NAME)
    log.info("  Lambda function created.")


def _get_latest_layer_version_arn(session, layer_name: str) -> str | None:
    """Return ARN of the latest published version of a Lambda layer."""
    lam = session.client("lambda", region_name=REGION)
    try:
        resp = lam.list_layer_versions(LayerName=layer_name, MaxItems=1)
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            return None
        raise
    versions = resp.get("LayerVersions") or []
    if not versions:
        return None
    return versions[0]["LayerVersionArn"]


def _masking_lambda_layer_arns(session) -> list[str]:
    """Ordered layer ARNs: extras (e.g. numpy) first, then ONNX runtime layer."""
    arns: list[str] = []
    for name in MASKING_EXTRA_LAYER_NAMES:
        arn = _get_latest_layer_version_arn(session, name)
        if not arn:
            log.error(
                "Masking extra layer %r has no published version. Publish it or remove "
                "the name from MASKING_EXTRA_LAYER_NAMES. "
                "See masking_lambda/build_numpy_layer.sh for a numpy layer.",
                name,
            )
            sys.exit(1)
        arns.append(arn)
    onnx_arn = _get_latest_layer_version_arn(session, MASKING_LAYER_NAME)
    if not onnx_arn:
        log.error(
            "No Lambda layer versions for %r in this account/region. "
            "Publish the ONNX layer first, e.g.: "
            "aws lambda publish-layer-version --layer-name %s --zip-file fileb://layer.zip ...",
            MASKING_LAYER_NAME,
            MASKING_LAYER_NAME,
        )
        sys.exit(1)
    arns.append(onnx_arn)
    return arns


def _zip_member_is_bundled_numpy(member_name: str) -> bool:
    """True if this zip path is a vendored numpy tree (duplicate when using numpy layer)."""
    n = member_name.replace("\\", "/")
    if n.startswith("numpy/") or n == "numpy":
        return True
    if n.startswith("numpy.libs/") or n == "numpy.libs":
        return True
    if "site-packages/numpy/" in n or "/site-packages/numpy/" in n:
        return True
    if "site-packages/numpy.libs/" in n:
        return True
    if "site-packages/numpy-" in n and ".dist-info/" in n:
        return True
    return False


def _prepare_masking_deployment_zip_for_upload() -> tuple[Path, bool]:
    """
    Return (path to zip, is_temp).

    When MASKING_STRIP_BUNDLED_NUMPY_FROM_ZIP and a numpy layer is used, remove numpy*
    from the deployment package so unzipped code + layers stays under 250 MB.
    """
    if not MASKING_DEPLOY_ZIP.is_file():
        log.error(
            "Masking deployment zip not found: %s\n"
            "Export the function package from the source account and place it there.",
            MASKING_DEPLOY_ZIP,
        )
        sys.exit(1)

    if not (
        MASKING_STRIP_BUNDLED_NUMPY_FROM_ZIP and MASKING_EXTRA_LAYER_NAMES
    ):
        return MASKING_DEPLOY_ZIP, False

    with zipfile.ZipFile(MASKING_DEPLOY_ZIP, "r") as zin:
        infos = zin.infolist()
        kept: list[zipfile.ZipInfo] = []
        dropped = 0
        for info in infos:
            if _zip_member_is_bundled_numpy(info.filename):
                dropped += 1
                continue
            kept.append(info)

        if dropped == 0:
            return MASKING_DEPLOY_ZIP, False

        log.info(
            "  Stripping %d bundled numpy file(s) from deployment.zip (supplied by layer).",
            dropped,
        )
        fd, tmp_path = tempfile.mkstemp(suffix=".zip", prefix="masking-deploy-")
        os.close(fd)
        tmp = Path(tmp_path)
        try:
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
                for info in kept:
                    data = zin.read(info.filename)
                    zout.writestr(info.filename, data, compress_type=zipfile.ZIP_DEFLATED)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

        return tmp, True


def _upload_masking_deployment_to_s3(session) -> None:
    """Stage deployment.zip in the pipeline bucket (supports large packages)."""
    zip_path, is_temp = _prepare_masking_deployment_zip_for_upload()
    try:
        s3 = session.client("s3", region_name=REGION)
        log.info(
            "  Uploading masking package → s3://%s/%s",
            S3_BUCKET,
            MASKING_S3_DEPLOY_KEY,
        )
        s3.upload_file(str(zip_path), S3_BUCKET, MASKING_S3_DEPLOY_KEY)
    finally:
        if is_temp:
            zip_path.unlink(missing_ok=True)


def ensure_masking_lambda(session, masking_role_arn: str) -> None:
    """Create or update the zip-based Aadhaar masking Lambda (uses S3-staged code)."""
    log.info("=== Step 6b: Ensure Aadhaar masking Lambda ===")

    _upload_masking_deployment_to_s3(session)
    layer_arns = _masking_lambda_layer_arns(session)

    lam = session.client("lambda", region_name=REGION)
    env_vars = _build_masking_lambda_env()
    code = {"S3Bucket": S3_BUCKET, "S3Key": MASKING_S3_DEPLOY_KEY}

    try:
        lam.get_function(FunctionName=MASKING_LAMBDA_NAME)
        log.info("  Masking Lambda exists, updating code + configuration …")

        lam.update_function_code(FunctionName=MASKING_LAMBDA_NAME, S3Bucket=S3_BUCKET, S3Key=MASKING_S3_DEPLOY_KEY)
        _wait_for_lambda_update(lam, MASKING_LAMBDA_NAME, max_wait=180)

        # Note: UpdateFunctionConfiguration does not accept Architectures (use create / code update to change arch).
        lam.update_function_configuration(
            FunctionName=MASKING_LAMBDA_NAME,
            Role=masking_role_arn,
            Handler=MASKING_LAMBDA_HANDLER,
            Runtime=MASKING_LAMBDA_RUNTIME,
            Timeout=MASKING_LAMBDA_TIMEOUT,
            MemorySize=MASKING_LAMBDA_MEMORY,
            EphemeralStorage={"Size": MASKING_LAMBDA_STORAGE},
            Layers=layer_arns,
            Environment={"Variables": env_vars},
        )
        _wait_for_lambda_update(lam, MASKING_LAMBDA_NAME, max_wait=180)
        _apply_masking_lambda_tags(lam)
        log.info("  Masking Lambda updated.")
        return

    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise

    log.info("  Creating masking Lambda: %s", MASKING_LAMBDA_NAME)
    lam.create_function(
        FunctionName=MASKING_LAMBDA_NAME,
        Runtime=MASKING_LAMBDA_RUNTIME,
        Role=masking_role_arn,
        Handler=MASKING_LAMBDA_HANDLER,
        Code=code,
        Timeout=MASKING_LAMBDA_TIMEOUT,
        MemorySize=MASKING_LAMBDA_MEMORY,
        EphemeralStorage={"Size": MASKING_LAMBDA_STORAGE},
        Architectures=[LAMBDA_ARCH],
        Layers=layer_arns,
        Environment={"Variables": env_vars},
        Tags=MASKING_LAMBDA_TAGS,
    )
    _wait_for_lambda_update(lam, MASKING_LAMBDA_NAME, max_wait=180)
    log.info("  Masking Lambda created.")


def _wait_for_lambda_update(lam_client, function_name: str, max_wait: int = 120):
    """Poll Lambda until LastUpdateStatus is Successful."""
    for _ in range(max_wait // 5):
        resp = lam_client.get_function_configuration(FunctionName=function_name)
        status = resp.get("LastUpdateStatus", "Successful")
        if status == "Successful":
            return
        if status == "Failed":
            raise RuntimeError(
                f"Lambda {function_name} update failed: {resp.get('LastUpdateStatusReason')}"
            )
        log.info("  Lambda %s status: %s — waiting …", function_name, status)
        time.sleep(5)
    raise RuntimeError(f"Lambda {function_name} update timed out")


# ── Step 7: Add S3 triggers ─────────────────────────────────────────────────


def _s3_notification_prefix_suffix(cfg: dict) -> tuple[str, str]:
    """Extract (prefix, suffix) from Lambda notification Filter; Name is case-insensitive."""
    prefix, suffix = "", ""
    for rule in cfg.get("Filter", {}).get("Key", {}).get("FilterRules", []) or []:
        name = (rule.get("Name") or "").lower()
        val = rule.get("Value") or ""
        if name == "prefix":
            prefix = val
        elif name == "suffix":
            suffix = val
    return prefix, suffix


def _lambda_notification_signature(cfg: dict) -> tuple:
    """Logical identity for deduping (console may use random Ids)."""
    arn = cfg.get("LambdaFunctionArn") or ""
    events = tuple(sorted(cfg.get("Events") or []))
    pre, suf = _s3_notification_prefix_suffix(cfg)
    return (arn, events, pre, suf)


def ensure_s3_triggers(session, include_masking_trigger: bool = True):
    """Add S3 triggers: pipeline_input/ → OCR Lambda; optionally chola_input/ → masking Lambda."""
    log.info("=== Step 7: Ensure S3 triggers ===")

    lam = session.client("lambda", region_name=REGION)
    s3 = session.client("s3", region_name=REGION)

    ocr_lambda_arn = f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:{LAMBDA_NAME}"
    masking_lambda_arn = f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:{MASKING_LAMBDA_NAME}"

    permission_pairs = [(LAMBDA_NAME, "s3-chola-pipeline-trigger")]
    if include_masking_trigger:
        permission_pairs.append((MASKING_LAMBDA_NAME, "s3-chola-masking-trigger"))

    for func_name, stmt_id in permission_pairs:
        try:
            lam.add_permission(
                FunctionName=func_name,
                StatementId=stmt_id,
                Action="lambda:InvokeFunction",
                Principal="s3.amazonaws.com",
                SourceArn=f"arn:aws:s3:::{S3_BUCKET}",
                SourceAccount=ACCOUNT_ID,
            )
            log.info("  Added S3 invoke permission for %s.", func_name)
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceConflictException":
                log.info("  S3 invoke permission for %s already exists.", func_name)
            else:
                raise

    log.info("  Waiting 5s for permission propagation …")
    time.sleep(5)

    # Get existing notification config
    existing = s3.get_bucket_notification_configuration(Bucket=S3_BUCKET)
    existing.pop("ResponseMetadata", None)

    lambda_configs = list(existing.get("LambdaFunctionConfigurations") or [])

    triggers = [
        {
            "Id": "OCRPipelineTrigger",
            "LambdaFunctionArn": ocr_lambda_arn,
            "Events": ["s3:ObjectCreated:*"],
            "Filter": {
                "Key": {"FilterRules": [{"Name": "prefix", "Value": "pipeline_input/"}]}
            },
        },
    ]
    if include_masking_trigger:
        triggers.append(
            {
                "Id": "AadhaarMaskingTrigger",
                "LambdaFunctionArn": masking_lambda_arn,
                "Events": ["s3:ObjectCreated:*"],
                "Filter": {
                    "Key": {"FilterRules": [{"Name": "prefix", "Value": "chola_input/"}]}
                },
            }
        )   

    # Drop any rule with the same logical target (ARN + events + prefix/suffix) so we do not
    # stack duplicates when the console created a rule with a random Id.
    managed_sigs = {_lambda_notification_signature(t) for t in triggers}
    before_n = len(lambda_configs)
    lambda_configs = [
        c
        for c in lambda_configs
        if _lambda_notification_signature(c) not in managed_sigs
    ]
    removed = before_n - len(lambda_configs)
    if removed:
        log.info(
            "  Replaced %d existing notification rule(s) with the same ARN/prefix (e.g. console UUID).",
            removed,
        )

    for trigger in triggers:
        lambda_configs.append(trigger)
        log.info("  Ensuring trigger: %s", trigger["Id"])

    existing["LambdaFunctionConfigurations"] = lambda_configs

    # Retry with backoff — S3 needs time to validate Lambda permissions
    for attempt in range(4):
        try:
            s3.put_bucket_notification_configuration(
                Bucket=S3_BUCKET,
                NotificationConfiguration=existing,
            )
            log.info("  S3 triggers configured successfully.")
            return
        except ClientError as e:
            if "Unable to validate" in str(e) and attempt < 3:
                wait = 5 * (attempt + 1)
                log.info("  Permission not yet propagated, retrying in %ds …", wait)
                time.sleep(wait)
            else:
                raise


# ── Helpers ──────────────────────────────────────────────────────────────────

def _run_cmd(cmd: list[str], input_data: str | None = None):
    """Run a shell command, raising on failure."""
    result = subprocess.run(
        cmd,
        input=input_data,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log.error("Command failed: %s", " ".join(cmd))
        log.error("stderr: %s", result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result.stdout


# ── Step 7b: Ensure DynamoDB table ──────────────────────────────────────────

def ensure_dynamodb_table(session) -> None:
    """Create DynamoDB table for pipeline logs if it doesn't exist."""
    log.info("=== Step 7b: Ensure DynamoDB table ===")

    dynamo = session.client("dynamodb", region_name=REGION)

    try:
        dynamo.describe_table(TableName=DYNAMO_TABLE)
        log.info("  DynamoDB table exists: %s", DYNAMO_TABLE)

        # Ensure TTL is enabled
        try:
            dynamo.update_time_to_live(
                TableName=DYNAMO_TABLE,
                TimeToLiveSpecification={"Enabled": True, "AttributeName": "ttl"},
            )
        except ClientError:
            pass  # Already enabled
        return
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise

    log.info("  Creating DynamoDB table: %s", DYNAMO_TABLE)
    dynamo.create_table(
        TableName=DYNAMO_TABLE,
        KeySchema=[{"AttributeName": "requestId", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "requestId", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",  # on-demand, no capacity planning needed
    )

    # Wait for table to become active
    waiter = dynamo.get_waiter("table_exists")
    waiter.wait(TableName=DYNAMO_TABLE)
    log.info("  Table active.")

    # Enable TTL for auto-expiry of old records
    dynamo.update_time_to_live(
        TableName=DYNAMO_TABLE,
        TimeToLiveSpecification={"Enabled": True, "AttributeName": "ttl"},
    )
    log.info("  TTL enabled on 'ttl' attribute (90-day auto-expiry).")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Deploy the OCR + Bedrock Lambda pipeline"
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="AWS CLI profile name (default: env/default)",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip uploading models to S3",
    )
    parser.add_argument(
        "--update-only",
        action="store_true",
        help="Only update Lambdas (skip models, bucket create, S3 triggers)",
    )
    parser.add_argument(
        "--skip-masking",
        action="store_true",
        help="Do not create/update the Aadhaar masking Lambda (requires zip + layer otherwise)",
    )
    args = parser.parse_args()

    # Build boto3 session
    session_kwargs = {}
    if args.profile:
        session_kwargs["profile_name"] = args.profile
    session = boto3.Session(**session_kwargs, region_name=REGION)

    # Auto-detect account ID and derive bucket name
    _resolve_account_id(session)

    log.info("Deploying to account %s in %s", ACCOUNT_ID, REGION)
    log.info("S3 bucket: %s (dedicated)", S3_BUCKET)

    if args.update_only:
        ecr_uri = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/{ECR_REPO}"
        image_uri = build_and_push_image(session, ecr_uri)
        role_arn = ensure_iam_role(session)   # refresh policy (e.g. new DynamoDB perms)
        ensure_dynamodb_table(session)
        ensure_lambda(session, image_uri, role_arn)
        if not args.skip_masking:
            masking_role_arn = ensure_masking_iam_role(session)
            ensure_masking_lambda(session, masking_role_arn)
        log.info("=== Update complete ===")
        return

    # Full deployment
    ensure_s3_bucket(session)

    if not args.skip_models:
        upload_models(session)

    ecr_uri = ensure_ecr_repo(session)
    image_uri = build_and_push_image(session, ecr_uri)
    role_arn = ensure_iam_role(session)
    ensure_dynamodb_table(session)
    ensure_lambda(session, image_uri, role_arn)
    if not args.skip_masking:
        masking_role_arn = ensure_masking_iam_role(session)
        ensure_masking_lambda(session, masking_role_arn)
    ensure_s3_triggers(session, include_masking_trigger=not args.skip_masking)

    log.info("=" * 60)
    log.info("Deployment complete!")
    log.info("  S3     : s3://%s/", S3_BUCKET)
    log.info("  Lambda : %s (%s)", LAMBDA_NAME, REGION)
    if not args.skip_masking:
        log.info("  Masking: %s (layer %s)", MASKING_LAMBDA_NAME, MASKING_LAYER_NAME)
    log.info("  Trigger: pipeline_input/ → OCR + Bedrock → pipeline_output/")
    log.info("  Trigger: chola_input/ → masking Lambda → chola_result/")
    log.info("  Models : models/paddleocr/ (OCR) + model/best_new.onnx (YOLO)")
    log.info("=" * 60)
    log.info(
        "Test: upload an image to s3://%s/pipeline_input/", S3_BUCKET
    )


if __name__ == "__main__":
    main()
