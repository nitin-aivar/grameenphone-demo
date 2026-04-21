# OCR + Document Extraction Lambda Pipeline — Deployment & Architecture Document

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [S3 Bucket Structure](#3-s3-bucket-structure)
4. [Lambda Configuration](#4-lambda-configuration)
5. [Code Implementation](#5-code-implementation)
6. [Deployment Process](#6-deployment-process)
7. [IAM Permissions](#7-iam-permissions)
8. [Cold Start & Warm Start Optimization](#8-cold-start--warm-start-optimization)
9. [End-to-End Latency Breakdown](#9-end-to-end-latency-breakdown)
10. [Cost Analysis](#10-cost-analysis)
11. [Troubleshooting & Fixes Applied](#11-troubleshooting--fixes-applied)
12. [Maintenance & Re-deployment](#12-maintenance--re-deployment)

---

## 1. Overview

This pipeline processes Indian identity documents (Aadhaar, PAN, Passport, Voter ID, E-Voter ID) uploaded to an S3 bucket. It performs:

1. **OCR** — Extracts text from images/PDFs using PaddleOCR (ONNX Runtime)
2. **Document Classification & Field Extraction** — Uses Amazon Bedrock (Claude 3 Haiku) to classify document type and extract structured fields
3. **Structured JSON Output** — Produces Chola/Karza-format JSON with all document fields
4. **Aadhaar Masking** — If Aadhaar is detected, automatically triggers an existing YOLO-based masking Lambda to redact sensitive information (QR codes, numbers)

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Lambda packaging | Container Image | `onnxruntime` (~180 MB) exceeds 250 MB ZIP limit |
| Region | ap-south-1 (Mumbai) | Co-located with S3 bucket and masking Lambda |
| Bedrock region | us-east-1 (cross-region) | Claude 3 Haiku availability; adds ~100 ms, negligible vs OCR time |
| Python version | 3.11 | Best `onnxruntime` wheel support |
| S3 bucket | Dedicated new bucket | Clean separation from existing masking infrastructure |

---

## 2. Architecture

```
                                  chola-ocr-pipeline-732592767494 (S3)
                                 ┌────────────────────────────────────┐
                                 │                                    │
  User uploads                   │  pipeline_input/                   │
  image/PDF ──────────────────►  │    Aadhar1.png                     │
                                 │    PAN1.png                        │
                                 │                                    │
                                 └──────────┬─────────────────────────┘
                                            │
                                   S3 Event Notification
                                   (s3:ObjectCreated:*)
                                   Prefix: pipeline_input/
                                            │
                                            ▼
                              ┌──────────────────────────────┐
                              │   chola-ocr-pipeline Lambda  │
                              │                              │
                              │  1. Download image from S3   │
                              │  2. Load PaddleOCR models    │
                              │     (from S3 → /tmp/models/) │
                              │  3. Run OCR (RapidOCR ONNX)  │
                              │  4. Call Bedrock Claude 3    │
                              │     Haiku for extraction     │
                              │  5. Save JSON output         │
                              │  6. If Aadhaar → copy to     │
                              │     chola_input/             │
                              └──────┬──────────┬────────────┘
                                     │          │
                           ┌─────────┘          └──────────┐
                           ▼                               ▼
                 pipeline_output/                   chola_input/
                   Aadhar1.json                     Aadhar1.png
                   PAN1.json                        (only if Aadhaar)
                                                           │
                                                  S3 Event Notification
                                                  Prefix: chola_input/
                                                           │
                                                           ▼
                                           ┌────────────────────────────┐
                                           │ equitas-aadhar-masking-test│
                                           │        Lambda             │
                                           │                           │
                                           │ YOLO ONNX masking model   │
                                           │ (model/best_new.onnx)     │
                                           └───────────┬───────────────┘
                                                       │
                                                       ▼
                                               chola_result/
                                             Aadhar1_masked.jpg
```

### Cross-Region Bedrock Access

```
Lambda (ap-south-1) ──── bedrock-runtime client ────► Bedrock (us-east-1)
                         region_name="us-east-1"       Claude 3 Haiku
                                                       (inference profile:
                                                        us.anthropic.claude-3-haiku-*)
```

The `us.` prefix model ID is a cross-region inference profile that can route to any US region (us-east-1, us-west-2, etc.) for load balancing.

---

## 3. S3 Bucket Structure

**Bucket:** `chola-ocr-pipeline-732592767494` (ap-south-1)

```
chola-ocr-pipeline-732592767494/
├── models/
│   └── paddleocr/
│       ├── det.onnx              # PaddleOCR detection model (~4.5 MB)
│       ├── rec.onnx              # PaddleOCR recognition model (~10 MB)
│       └── dict.txt              # Character dictionary (~200 KB)
├── model/
│   └── best_new.onnx             # YOLO masking model (~42.7 MB)
├── pipeline_input/               # TRIGGER: Upload documents here
├── pipeline_output/              # OUTPUT: Structured JSON results
├── chola_input/                  # TRIGGER: Aadhaar images for masking
└── chola_result/                 # OUTPUT: Masked Aadhaar images
```

### S3 Event Notifications

| Trigger ID | Prefix | Target Lambda |
|------------|--------|---------------|
| `OCRPipelineTrigger` | `pipeline_input/` | `chola-ocr-pipeline` |
| `AadhaarMaskingTrigger` | `chola_input/` | `equitas-aadhar-masking-test` |

---

## 4. Lambda Configuration

### OCR Pipeline Lambda: `chola-ocr-pipeline`

| Setting | Value | Rationale |
|---------|-------|-----------|
| **Function name** | `chola-ocr-pipeline` | |
| **Runtime** | Container image (Python 3.11) | onnxruntime exceeds ZIP limit |
| **Architecture** | x86_64 | Best onnxruntime wheel support |
| **Memory** | 3008 MB | OCR needs ~1.5–2.5 GB + PDF rasterization |
| **Timeout** | 300 seconds (5 min) | OCR ~25–35s + Bedrock ~5–10s; covers multi-page PDFs |
| **Ephemeral storage (/tmp)** | 1024 MB | Models (~15 MB) + input files + poppler temp |
| **Reserved concurrency** | Not set (default) | Adjust based on traffic |
| **ECR repository** | `chola-ocr-pipeline` | ap-south-1 |
| **IAM role** | `chola-ocr-pipeline-lambda-role` | |

### Environment Variables

| Variable | Value | Description |
|----------|-------|-------------|
| `S3_BUCKET` | `chola-ocr-pipeline-732592767494` | Target bucket for outputs |
| `MODEL_BUCKET` | `chola-ocr-pipeline-732592767494` | Bucket containing OCR models |
| `MODEL_PREFIX` | `models/paddleocr` | S3 prefix for PaddleOCR models |
| `OUTPUT_PREFIX` | `pipeline_output` | S3 prefix for JSON output |
| `MASKING_PREFIX` | `chola_input` | S3 prefix for Aadhaar masking trigger |
| `BEDROCK_MODEL` | `us.anthropic.claude-3-haiku-20240307-v1:0` | Bedrock model ID |
| `BEDROCK_REGION` | `us-east-1` | Bedrock endpoint region |

---

## 5. Code Implementation

### File Structure

```
lambda/
├── handler.py           # 150 lines — Lambda entry point
├── ocr_engine.py        # 180 lines — PaddleOCR ONNX engine
├── json_formatter.py    # 424 lines — Bedrock extraction + Verhoeff
├── Dockerfile           # 13 lines  — Container image definition
├── requirements.txt     # 5 lines   — Python dependencies
├── iam-policy.json      # 35 lines  — Reference IAM policy
├── deploy.py            # 579 lines — Automated deployment script
└── test_local.py        # Local testing script
```

### handler.py — Lambda Entry Point

**Function:** `lambda_handler(event, context)`

**Flow:**
1. Parse S3 event → extract bucket name and object key
2. Validate file extension (images: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif`, `.webp`; PDFs: `.pdf`)
3. Download input file from S3 to `/tmp/`
4. Initialize OCR engine (cached across warm invocations)
5. Run OCR → raw result with text blocks, bounding boxes, confidence scores
6. Call Bedrock via `format_ocr_result()` → structured document list
7. Build output envelope: `{ requestId, result: { documents }, statusCode: 101 }`
8. Save JSON to `s3://bucket/pipeline_output/{stem}.json`
9. If any document is `AADHAAR` → `copy_object` to `chola_input/` (server-side copy, no re-upload)
10. Clean up `/tmp/` input file

**Module-level clients** (reused across warm invocations):
```python
s3_client = boto3.client("s3")
bedrock_client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
```

### ocr_engine.py — OCR Engine

**Key functions:**

| Function | Purpose |
|----------|---------|
| `get_engine(s3_client, bucket, prefix)` | Returns cached engine or builds new one |
| `_download_models_from_s3(...)` | Downloads det.onnx, rec.onnx, dict.txt to `/tmp/models/` |
| `_build_engine(paths)` | Creates RapidOCR instance with custom model paths |
| `ocr_image(engine, img_input)` | Runs OCR on image/PIL.Image → list of blocks |
| `process_image(engine, path)` | Single image → structured result dict |
| `process_pdf(engine, path)` | Multi-page PDF → structured result dict |

**Warm start optimization:**
- Module-level global `_engine: RapidOCR | None = None`
- On cold start: download models from S3 (~1.5s), initialize ONNX sessions
- On warm start: skip download, reuse cached engine from memory + `/tmp/`

**Config compatibility fix:**
The Docker image installs a newer version of `rapidocr-onnxruntime` where:
- Package directory is found via `Path(rapidocr_onnxruntime.__file__).parent` (not `rapid_ocr_api.root_dir`)
- Character dictionary config key is `rec_keys_path` (not `keys_path`)

### json_formatter.py — Bedrock Extraction

**Key functions:**

| Function | Purpose |
|----------|---------|
| `format_ocr_result(bedrock_client, ocr_result, model_id)` | Main entry point |
| `call_bedrock(client, user_message, model_id)` | Calls Bedrock Converse API with 3 retries |
| `extract_json_from_text(text)` | Robust JSON extraction (handles fences, truncation) |
| `enrich_documents(documents)` | Adds Verhoeff checksum, mask status, address split |
| `verhoeff_checksum(number_str)` | Validates Aadhaar numbers using Verhoeff algorithm |

**Supported document types:**
- AADHAAR (front + back sections)
- PAN
- PASSPORT
- VOTER_ID
- E_VOTER_ID

**System prompt:** 100+ lines of detailed instructions for Claude 3 Haiku, including:
- Schema definitions per document type
- Field extraction rules
- Multi-page document handling
- Edge case instructions

### Dockerfile

```dockerfile
FROM public.ecr.aws/lambda/python:3.11

# poppler-utils required for pdf2image (PDF page rasterization)
RUN yum install -y poppler-utils && yum clean all

COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

COPY handler.py ocr_engine.py json_formatter.py ${LAMBDA_TASK_ROOT}/
CMD ["handler.lambda_handler"]
```

### requirements.txt

```
numpy<2.0                # Pinned: numpy 2.x has no prebuilt wheel for Lambda's Python 3.11 x86_64
rapidocr-onnxruntime     # RapidOCR with ONNX Runtime backend (~180 MB with onnxruntime)
Pillow>=10.0             # Image processing
pdf2image>=1.16          # PDF → PIL Image conversion (requires poppler-utils)
PyYAML>=6.0              # YAML config parsing for RapidOCR
```

> **Note:** `boto3` is NOT listed — it's pre-installed in the Lambda Python runtime.

---

## 6. Deployment Process

### Prerequisites

- AWS CLI configured with SSO profile
- Docker Desktop running
- Python 3.11+ with boto3 installed locally

### One-Command Deployment

```bash
cd lambda/
python deploy.py --profile my-sso-profile
```

### What `deploy.py` Does (Step by Step)

#### Step 1: Create S3 Bucket
```
Creates: chola-ocr-pipeline-732592767494 (ap-south-1)
- Public access blocked
- Skip if already exists
```

#### Step 2: Upload Models to S3
```
PaddleOCR models (from HuggingFace → S3):
  monkt/paddleocr-onnx → models/paddleocr/det.onnx
  monkt/paddleocr-onnx → models/paddleocr/rec.onnx
  monkt/paddleocr-onnx → models/paddleocr/dict.txt

YOLO masking model (from local → S3):
  yolo_masking_model/best_new.onnx → model/best_new.onnx
```

#### Step 3: Create ECR Repository
```
Repository: chola-ocr-pipeline (ap-south-1)
Skip if already exists
```

#### Step 4: Build & Push Docker Image
```bash
# Authenticate Docker to ECR (via boto3, not AWS CLI)
docker login --username AWS --password-stdin 732592767494.dkr.ecr.ap-south-1.amazonaws.com

# Build for Linux x86_64 (required for Lambda)
docker buildx build --platform linux/amd64 --provenance=false -t <ecr-uri>:latest .

# Push to ECR
docker push <ecr-uri>:latest
```

> **Important:** `--provenance=false` is required on Apple Silicon Macs. Without it, Docker creates OCI attestation manifests that Lambda rejects.

#### Step 5: Create IAM Role & Policy
```
Role: chola-ocr-pipeline-lambda-role
Trust policy: lambda.amazonaws.com
Inline policy: chola-ocr-pipeline-policy (see IAM Permissions section)
```

#### Step 6: Create/Update Lambda Function
```
Function: chola-ocr-pipeline
Image: 732592767494.dkr.ecr.ap-south-1.amazonaws.com/chola-ocr-pipeline:latest
Memory: 3008 MB | Timeout: 300s | /tmp: 1024 MB
Environment variables set
```

#### Step 7: Configure S3 Triggers
```
Notification 1: pipeline_input/ → chola-ocr-pipeline
Notification 2: chola_input/ → equitas-aadhar-masking-test
Includes retry logic (5s + exponential backoff) for Lambda permission propagation
```

### Incremental Deployment

```bash
# Skip model upload (re-deploy code only)
python deploy.py --profile my-sso-profile --skip-models

# Update Lambda image only (fastest)
python deploy.py --profile my-sso-profile --update-only
```

---

## 7. IAM Permissions

**Role:** `chola-ocr-pipeline-lambda-role`
**Policy:** `chola-ocr-pipeline-policy`

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "S3ReadWrite",
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:PutObject"],
            "Resource": "arn:aws:s3:::chola-ocr-pipeline-732592767494/*"
        },
        {
            "Sid": "BedrockInvoke",
            "Effect": "Allow",
            "Action": ["bedrock:InvokeModel"],
            "Resource": [
                "arn:aws:bedrock:*::foundation-model/anthropic.claude-3-haiku-*",
                "arn:aws:bedrock:*:732592767494:inference-profile/us.anthropic.claude-3-haiku-*"
            ]
        },
        {
            "Sid": "CloudWatchLogs",
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "arn:aws:logs:ap-south-1:732592767494:*"
        }
    ]
}
```

> **Note:** Bedrock resources use wildcard `*` for region because the cross-region inference profile (`us.anthropic.claude-3-haiku-*`) can route requests to any US region (us-east-1, us-west-2, etc.).

---

## 8. Cold Start & Warm Start Optimization

### Cold Start (First Invocation)

| Phase | Duration | What Happens |
|-------|----------|-------------|
| Container init | ~2.0s | Lambda pulls container, initializes Python runtime |
| Module imports | ~0.5s | Import boto3, rapidocr, PIL, numpy |
| S3 model download | ~1.5s | Download det.onnx + rec.onnx + dict.txt from same-region S3 |
| ONNX session init | ~0.5s | Build RapidOCR engine, load ONNX sessions |
| **Total cold start** | **~4.5s** | Added to first invocation only |

### Warm Start (Subsequent Invocations)

| What Persists | How |
|--------------|-----|
| Python runtime | Lambda container reused |
| boto3 clients | Module-level `s3_client`, `bedrock_client` |
| OCR engine | Module-level `_engine` global |
| Model files | `/tmp/models/` persists between invocations |
| **Warm start overhead** | **~0s** (no init needed) |

---

## 9. End-to-End Latency Breakdown

Latency data captured from real Lambda invocations during testing (CloudWatch logs).

### Single Image Processing (Warm Start)

| Step | Aadhaar Image | PAN Image | Notes |
|------|--------------|-----------|-------|
| S3 download (input) | ~0.1s | ~0.1s | Same-region, small file |
| OCR (RapidOCR ONNX) | **24.6–25.4s** | **29.9–32.7s** | Main bottleneck |
| Bedrock (Claude 3 Haiku) | **~5–8s** | **~3–5s** | Cross-region to us-east-1 |
| S3 upload (JSON output) | ~0.1s | ~0.1s | Small JSON file |
| S3 copy (Aadhaar → masking) | ~0.1s | N/A | Server-side copy |
| **Total (warm start)** | **~30–34s** | **~33–38s** | |
| **Total (cold start)** | **~35–39s** | **~38–43s** | +4.5s init |

### Masking Lambda (Aadhaar Only)

| Step | Duration | Notes |
|------|----------|-------|
| YOLO model load | ~2–3s (cold) / ~0s (warm) | Cached in memory |
| Image masking | ~3–5s | QR code + number detection + blur |
| **Total masking** | **~5–8s** | |

### Complete End-to-End (Upload → Masked Output)

| Scenario | Total Latency |
|----------|---------------|
| PAN card (no masking) | **~33–38s** |
| Aadhaar (with masking) | **~38–47s** |
| PDF (multi-page, 3 pages) | **~90–120s** (estimated) |

### Latency Distribution

```
                       OCR (70%)          Bedrock (15%)
    ├──────────────────────────────────┤────────────┤──┤
    0s           10s           20s           30s    35s
                                                    │
                                               S3 I/O (2%)
                                               Init (13% cold only)
```

> **OCR is the dominant bottleneck** at ~70% of total execution time. Bedrock is ~15%. S3 operations are negligible.

### Memory Usage (from CloudWatch REPORT)

| Image Type | Max Memory Used | Allocated |
|------------|----------------|-----------|
| Aadhaar | 1,755 MB | 3,008 MB |
| PAN | 2,087–2,430 MB | 3,008 MB |

---

## 10. Cost Analysis

### Unit Prices (ap-south-1 / us-east-1 for Bedrock)

> **Source:** Prices verified via AWS Price List API (`awslabs.aws-pricing-mcp-server`) on 2026-03-24.
> Publication dates: Lambda 2026-03-16, S3 2026-02-23, Bedrock 2026-03-23, ECR 2025-11-21.

| Service | Metric | Price |
|---------|--------|-------|
| **Lambda compute** | per GB-second | $0.0000166667 |
| **Lambda requests** | per 1M requests | $0.20 |
| **S3 storage** | per GB/month | $0.025 |
| **S3 PUT/COPY** | per 1,000 requests | $0.005 |
| **S3 GET** | per 1,000 requests | $0.0004 |
| **Bedrock Claude 3 Haiku (input)** | per 1M tokens | $0.25 |
| **Bedrock Claude 3 Haiku (output)** | per 1M tokens | $1.25 |
| **ECR storage** | per GB/month | $0.10 |
| **CloudWatch Logs** | per GB ingested | $0.67 |

### Per-Invocation Cost Breakdown

**Assumptions for one image (single page):**
- Lambda duration: 35 seconds (average)
- Lambda memory: 3,008 MB = 2.9375 GB
- Bedrock input tokens: ~2,500 tokens (system prompt + OCR text)
- Bedrock output tokens: ~800 tokens (structured JSON response)
- S3 operations: 1 GET (input) + 1 PUT (JSON) + 1 COPY (if Aadhaar) = 2–3 operations
- CloudWatch log: ~2 KB per invocation

| Component | Calculation | Cost per Invocation |
|-----------|------------|---------------------|
| **Lambda compute** | 35s × 2.9375 GB × $0.0000166667/GB-s | **$0.001713** |
| **Lambda request** | 1 × $0.0000002 | **$0.0000002** |
| **Bedrock input** | 2,500 tokens × $0.00000025/token | **$0.000625** |
| **Bedrock output** | 800 tokens × $0.00000125/token | **$0.001000** |
| **S3 requests** | 3 operations ≈ $0.005/1000 × 3 | **$0.000015** |
| **CloudWatch** | 2 KB × $0.67/GB | **$0.0000013** |
| **TOTAL** | | **$0.003354** |

> **Average cost per document: ~$0.0034 (approximately $0.34 per 100 documents)**

### Monthly Cost at Scale: 350,000 Pages/Month

| Component | Calculation | Monthly Cost |
|-----------|------------|-------------|
| **Lambda compute** | 350,000 × 35s × 2.9375 GB × $0.0000166667 | **$599.56** |
| **Lambda requests** | 350,000 × $0.0000002 | **$0.07** |
| **Bedrock input tokens** | 350,000 × 2,500 × $0.00000025 | **$218.75** |
| **Bedrock output tokens** | 350,000 × 800 × $0.00000125 | **$350.00** |
| **S3 requests** | 350,000 × 3 × $0.005/1000 | **$5.25** |
| **S3 storage** | ~5 GB (JSON outputs + images) × $0.025 | **$0.13** |
| **ECR storage** | ~1.5 GB (container image) × $0.10 | **$0.15** |
| **CloudWatch Logs** | 350,000 × 2 KB ≈ 0.7 GB × $0.67 | **$0.47** |
| **TOTAL** | | **$1,174.38/month** |

### Cost Breakdown Visualization (350K pages/month)

```
Lambda compute    ████████████████████████████████  $599.56  (51.1%)
Bedrock output    ██████████████████               $350.00  (29.8%)
Bedrock input     ███████████                      $218.75  (18.6%)
S3 requests       ▌                                  $5.25  (0.4%)
Other             ▏                                  $0.82  (0.1%)
                  ─────────────────────────────────
                  TOTAL                           $1,174.38
```

### Free Tier Savings (First 12 months)

| Service | Free Tier | Monthly Savings |
|---------|-----------|----------------|
| Lambda | 1M requests + 400,000 GB-s | ~$6.67 (compute) |
| S3 | 5 GB storage + 20,000 GETs + 2,000 PUTs | ~$0.15 |
| ECR | 500 MB storage | ~$0.05 |
| **Total free tier savings** | | **~$6.87/month** |

### Cost per Document at Different Scales

| Monthly Volume | Cost/Month | Cost/Document |
|---------------|-----------|---------------|
| 1,000 | $3.35 | $0.00335 |
| 10,000 | $33.54 | $0.00335 |
| 100,000 | $335.40 | $0.00335 |
| 350,000 | $1,174.38 | $0.00336 |
| 1,000,000 | $3,354.00 | $0.00335 |

> Cost scales linearly. There are no step-function pricing tiers at these volumes.

### Cost Optimization Opportunities

| Optimization | Potential Savings | Trade-off |
|-------------|------------------|-----------|
| **Reduce Lambda memory to 2048 MB** | ~32% on compute ($192/mo at 350K) | May increase OCR time, risk OOM for large PDFs |
| **Bake models into container image** | Eliminates ~1.5s cold start download | Larger ECR image (+15 MB), slower builds |
| **Use Provisioned Concurrency** | Eliminates cold start entirely | Fixed cost regardless of traffic |
| **Batch processing** | Process multiple pages per invocation | More complex handler logic |
| **Switch to Claude 3.5 Haiku** | Different pricing, potentially faster | May require prompt tuning |

---

## 11. Troubleshooting & Fixes Applied

Issues encountered during deployment and their fixes:

### Issue 1: numpy Build Failure in Docker
- **Error:** `numpy 2.4.3` had no prebuilt wheel for Python 3.11 x86_64, tried to compile from source (no gcc in Lambda image)
- **Fix:** Pinned `numpy<2.0` in requirements.txt to use prebuilt 1.26.x wheels

### Issue 2: Docker Provenance Manifest Rejected
- **Error:** `docker buildx build --platform linux/amd64` on Apple Silicon creates OCI attestation manifests that Lambda doesn't support
- **Fix:** Added `--provenance=false` to the Docker buildx command

### Issue 3: rapidocr API Change
- **Error:** `ModuleNotFoundError: No module named 'rapidocr_onnxruntime.rapid_ocr_api'`
- **Fix:** Replaced `from rapidocr_onnxruntime.rapid_ocr_api import root_dir` with `Path(rapidocr_onnxruntime.__file__).parent`

### Issue 4: Character Dictionary Config Key
- **Error:** `ValueError: character must not be None` — recognizer couldn't find dictionary
- **Fix:** Changed config key from `keys_path` to `rec_keys_path` (newer rapidocr version)

### Issue 5: Bedrock Cross-Region IAM
- **Error:** `AccessDeniedException` — IAM policy only allowed `us-east-1` ARN, but cross-region inference profile routes to any US region
- **Fix:** Changed Bedrock resource ARNs to use wildcard `*` for region

### Issue 6: S3 Trigger Permission Propagation
- **Error:** `put_bucket_notification_configuration` failed with "Unable to validate" immediately after adding Lambda permission
- **Fix:** Added 5-second initial wait + retry loop with exponential backoff (3 attempts)

---

## 12. Maintenance & Re-deployment

### Updating Lambda Code

```bash
cd lambda/

# 1. Edit the Python files
# 2. Rebuild and push
docker buildx build --platform linux/amd64 --provenance=false \
  -t 732592767494.dkr.ecr.ap-south-1.amazonaws.com/chola-ocr-pipeline:latest .
docker push 732592767494.dkr.ecr.ap-south-1.amazonaws.com/chola-ocr-pipeline:latest

# 3. Update Lambda
python deploy.py --profile my-sso-profile --update-only
```

### Updating Models

```bash
# Re-upload models to S3
python deploy.py --profile my-sso-profile
# (Default: uploads all models)
```

### Monitoring

- **CloudWatch Logs:** `/aws/lambda/chola-ocr-pipeline`
- **Key metrics to watch:**
  - Duration (should be 30–40s for single images)
  - Max Memory Used (should stay under 3,008 MB)
  - Error count (Bedrock throttling, timeouts)
  - Invocation count

### Cleanup (Old Resources)

The following resources from earlier iterations should be removed:

| Resource | Location | Action |
|----------|----------|--------|
| PaddleOCR models | `s3://equitas-aadhar-masking-testing/models/paddleocr/` | Delete folder |
| Test files | `s3://equitas-aadhar-masking-testing/pipeline_input/` | Delete folder |
| Test outputs | `s3://equitas-aadhar-masking-testing/pipeline_output/` | Delete folder |
| S3 notification | `equitas-aadhar-masking-testing` → `pipeline_input/` trigger | Remove notification |
| Docker images | Local machine | `docker image prune` |

---

## Appendix: AWS Resources Created

| Resource | Type | Region | Name/ID |
|----------|------|--------|---------|
| S3 Bucket | Storage | ap-south-1 | `chola-ocr-pipeline-732592767494` |
| ECR Repository | Container Registry | ap-south-1 | `chola-ocr-pipeline` |
| Lambda Function | Compute | ap-south-1 | `chola-ocr-pipeline` |
| IAM Role | Security | Global | `chola-ocr-pipeline-lambda-role` |
| IAM Policy | Security | Global | `chola-ocr-pipeline-policy` (inline) |
| CloudWatch Log Group | Logging | ap-south-1 | `/aws/lambda/chola-ocr-pipeline` |
| Lambda Permission | Security | ap-south-1 | S3 → Lambda invoke permission |
