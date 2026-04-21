import json
import os
import re
import time
import uuid
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

BUCKET = os.environ.get("S3_BUCKET", "chola-ocr-pipeline-732592767494")
REGION = "ap-south-1"
INPUT_PREFIX = "pipeline_input"
OUTPUT_PREFIX = "pipeline_output"
MASKED_PREFIX = "chola_result"
PAGES_PREFIX = "pipeline_pages"


def get_s3_client():
    return boto3.client("s3", region_name=REGION)


def _safe_stem_and_filename(original_filename: str) -> tuple[str, str]:
    """
    Basename + unique stem so each upload gets its own S3 key.
    Prevents stale cached results when re-uploading the same file
    with different settings (e.g. mask_digits).
    """
    base_name = Path(original_filename).name
    if not base_name or base_name in (".", ".."):
        base_name = "document"
    stem_raw, ext = os.path.splitext(base_name)
    ext = ext.lower().lstrip(".")
    stem = re.sub(r"[^\w\-]", "_", stem_raw or "document")
    unique_stem = f"{uuid.uuid4().hex[:8]}_{stem}"
    s3_filename = f"{unique_stem}.{ext}" if ext else unique_stem
    return s3_filename, unique_stem


def upload_file(file_bytes: bytes, original_filename: str, mask_digits: int = 8) -> str:
    """
    Upload a file to S3 pipeline_input/{sanitized_basename}.
    Returns the stem used for polling (same stem as masking output prefix).
    mask_digits is stored as S3 user-metadata so the masking Lambda can read it.
    """
    s3_filename, stem = _safe_stem_and_filename(original_filename)
    s3_key = f"{INPUT_PREFIX}/{s3_filename}"

    ext = s3_filename.rsplit(".", 1)[-1].lower() if "." in s3_filename else ""
    content_type_map = {
        "pdf": "application/pdf",
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
    }
    content_type = content_type_map.get(ext, "application/octet-stream")

    s3 = get_s3_client()
    s3.put_object(
        Bucket=BUCKET,
        Key=s3_key,
        Body=file_bytes,
        ContentType=content_type,
        Metadata={"mask_digits": str(mask_digits)},
        ServerSideEncryption="AES256",
    )

    return stem


def poll_json_result(stem: str, timeout: int = 300, interval: int = 5):
    """
    Poll pipeline_output/{stem}.json until it exists or timeout is reached.
    Returns parsed JSON dict on success, None on timeout.
    """
    s3 = get_s3_client()
    s3_key = f"{OUTPUT_PREFIX}/{stem}.json"
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            response = s3.get_object(Bucket=BUCKET, Key=s3_key)
            return json.loads(response["Body"].read())
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                time.sleep(interval)
            else:
                raise
    return None


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def fetch_masked_image(stem: str, timeout: int = 120, interval: int = 5) -> dict[int, bytes] | None:
    """
    Poll chola_result/ for masked output from the masking Lambda.

    Handles two naming patterns:
      - Single image:  chola_result/{stem}_masked.jpg|pdf
      - PDF pages:     chola_result/{stem}_p1_masked.jpg, …_p2_masked.jpg, …

    Returns a dict mapping page number (1-based) to raw image/pdf bytes,
    or None on timeout.
    """
    s3 = get_s3_client()
    deadline = time.time() + timeout

    single_prefix = f"{MASKED_PREFIX}/{stem}_masked."
    page_prefix = f"{MASKED_PREFIX}/{stem}_p"

    while time.time() < deadline:
        result: dict[int, bytes] = {}
        try:
            # 1) Check single-image pattern: {stem}_masked.*
            resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=single_prefix, MaxKeys=10)
            image_keys: list[str] = []
            pdf_keys: list[str] = []
            for obj in resp.get("Contents", []):
                key = obj["Key"]
                base = key.rsplit("/", 1)[-1]
                if not base.startswith(f"{stem}_masked."):
                    continue
                ext = "." + key.rsplit(".", 1)[-1].lower() if "." in key else ""
                if ext in _IMAGE_EXTS:
                    image_keys.append(key)
                elif ext == ".pdf":
                    pdf_keys.append(key)

            pick = None
            if image_keys:
                pick = sorted(image_keys)[0]
            elif pdf_keys:
                pick = sorted(pdf_keys)[0]
            if pick:
                data = s3.get_object(Bucket=BUCKET, Key=pick)["Body"].read()
                return {1: data}

            # 2) Check per-page pattern: {stem}_p{N}_masked.*
            resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=page_prefix, MaxKeys=50)
            import re as _re
            page_pattern = _re.compile(
                rf"^{_re.escape(stem)}_p(\d+)_masked\.\w+$"
            )
            for obj in resp.get("Contents", []):
                key = obj["Key"]
                base = key.rsplit("/", 1)[-1]
                m = page_pattern.match(base)
                if not m:
                    continue
                ext = "." + key.rsplit(".", 1)[-1].lower() if "." in key else ""
                if ext not in _IMAGE_EXTS:
                    continue
                page_no = int(m.group(1))
                data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
                result[page_no] = data

            if result:
                return result

        except ClientError:
            pass
        time.sleep(interval)
    return None


def fetch_page_image(stem: str, page_no: int) -> bytes | None:
    """
    Fetch the JPEG bytes for a specific page image saved by the Lambda.
    Returns raw bytes or None if not found.
    """
    s3 = get_s3_client()
    s3_key = f"{PAGES_PREFIX}/{stem}_p{page_no}.jpg"
    try:
        response = s3.get_object(Bucket=BUCKET, Key=s3_key)
        return response["Body"].read()
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def is_aadhaar_result(result_json: dict) -> bool:
    """Check if any document in the result is AADHAAR type."""
    docs = result_json.get("result", {}).get("documents", [])
    return any(d.get("documentType", "").upper() == "AADHAAR" for d in docs)
