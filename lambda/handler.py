"""
AWS Lambda handler: OCR + Mistral vision document extraction pipeline.

Triggered by S3 uploads to pipeline_input/ prefix. Runs PaddleOCR on the
uploaded image/PDF, classifies document type via regex, triggers early
Aadhaar masking if detected, calls Mistral Ministral 3 14B vision model
for structured field extraction, saves JSON to pipeline_output/, and
falls back to masking if Aadhaar is found in the final JSON.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote_plus

import boto3

from ocr_engine import get_engine, process_image, process_pdf, IMAGE_EXTS, PDF_EXTS
from json_formatter import format_ocr_result
from doc_classifier import classify_document
import dynamo_logger as db

# ── Configuration (via environment variables) ────────────────────────────────

S3_BUCKET = os.environ.get("S3_BUCKET", "chola-ocr-pipeline-757333951934")
OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "pipeline_output")
MASKING_PREFIX = os.environ.get("MASKING_PREFIX", "chola_input")
BEDROCK_MODEL = os.environ.get(
    "BEDROCK_MODEL", "mistral.ministral-3-14b-instruct"
)
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "ap-south-1")
MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "chola-ocr-pipeline-757333951934")
MODEL_PREFIX = os.environ.get("MODEL_PREFIX", "models/paddleocr")

# ── Logging (with PII redaction) ─────────────────────────────────────────────

import re as _re

_PII_PATTERNS = [
    (_re.compile(r"\b(\d{4})\s?(\d{4})\s?(\d{4})\b"), r"XXXX XXXX \3"),  # Aadhaar: mask first 8 digits
    (_re.compile(r"\b(\d{12})\b"), lambda m: "XXXXXXXX" + m.group(1)[-4:]),  # 12-digit continuous
    (_re.compile(r"(?i)(vid\s*:?\s*)\d{16}"), r"\1XXXXXXXXXXXXXXXX"),  # VID
]


class _PIIRedactFilter(logging.Filter):
    """Redact Aadhaar numbers and VIDs from log messages.

    Formats the message early (msg % args) so PII in args is caught,
    then clears args so the logging framework doesn't double-format.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        # Format the message first so PII in args gets included
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)
        for pattern, repl in _PII_PATTERNS:
            msg = pattern.sub(repl, msg)
        record.msg = msg
        record.args = None
        return True


log = logging.getLogger()
log.setLevel(logging.INFO)
log.addFilter(_PIIRedactFilter())

# ── Clients (module-level for connection reuse across warm invocations) ──────

s3_client = boto3.client("s3")
bedrock_client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)


# ── Handler ──────────────────────────────────────────────────────────────────

def lambda_handler(event, context):
    """Main Lambda entry point — S3 event trigger."""

    pipeline_start = time.monotonic()

    # 1. Parse S3 event
    record = event["Records"][0]
    bucket = record["s3"]["bucket"]["name"]
    key = unquote_plus(record["s3"]["object"]["key"])
    filename = key.split("/")[-1]

    if "." not in filename:
        log.error("No file extension in key: %s", key)
        return {"statusCode": 400, "body": "Unsupported file (no extension)"}

    stem = filename.rsplit(".", 1)[0]
    ext = "." + filename.rsplit(".", 1)[1].lower()

    if ext not in IMAGE_EXTS and ext not in PDF_EXTS:
        log.warning("Unsupported file type: %s", ext)
        return {"statusCode": 400, "body": f"Unsupported file type: {ext}"}

    # 1a. Guard: only process files from expected input prefix
    if not key.startswith("pipeline_input/"):
        log.warning("Ignoring key outside pipeline_input/: %s", key)
        return {"statusCode": 200, "body": "Ignored — not in pipeline_input/"}

    log.info("Processing s3://%s/%s", bucket, key)

    # Generate requestId upfront so DynamoDB record exists from the start
    request_id = str(uuid.uuid4())
    file_type = "pdf" if ext in PDF_EXTS else "image"

    # Create the initial DynamoDB record (status=PROCESSING)
    db.create_record(request_id, key, filename, file_type, ext)

    local_path = Path(f"/tmp/{uuid.uuid4().hex}_{filename}")

    try:
        # 2. Download input file from S3 + read mask_digits metadata
        t0 = time.monotonic()
        s3_client.download_file(bucket, key, str(local_path))
        download_ms = int((time.monotonic() - t0) * 1000)
        log.info("[LATENCY] s3_download=%dms (%s)", download_ms, filename)
        db.update_step(request_id, "download", download_ms)

        head = s3_client.head_object(Bucket=bucket, Key=key)
        mask_digits = head.get("Metadata", {}).get("mask_digits", "8")
        log.info("mask_digits=%s (from upload metadata)", mask_digits)

        # 3. Get OCR engine (cached across warm starts)
        t0 = time.monotonic()
        engine = get_engine(s3_client, MODEL_BUCKET, MODEL_PREFIX)
        engine_ms = int((time.monotonic() - t0) * 1000)
        log.info("[LATENCY] engine_init=%dms", engine_ms)

        # ── IMAGE path ────────────────────────────────────────────────────────
        if ext not in PDF_EXTS:
            t0 = time.monotonic()
            ocr_result = process_image(engine, local_path)
            ocr_ms = int((time.monotonic() - t0) * 1000)
            num_pages = 1
            total_blocks = ocr_result["total_blocks"]
            log.info("[LATENCY] ocr=%dms (%d blocks)", ocr_ms, total_blocks)

            t0 = time.monotonic()
            image_bytes_list = ocr_result.get("image_bytes_list", [])
            for idx, page_img_bytes in enumerate(image_bytes_list, 1):
                page_key = f"pipeline_pages/{stem}_p{idx}.jpg"
                try:
                    s3_client.put_object(Bucket=S3_BUCKET, Key=page_key, Body=page_img_bytes, ContentType="image/jpeg")
                except Exception as exc:
                    log.warning("Failed to save page image %d: %s", idx, exc)
            page_upload_ms = int((time.monotonic() - t0) * 1000)
            log.info("[LATENCY] page_images_upload=%dms", page_upload_ms)

            t0 = time.monotonic()
            doc_type = classify_document(ocr_result["full_text"])
            classify_ms = int((time.monotonic() - t0) * 1000)
            log.info("[LATENCY] classification=%dms → %s", classify_ms, doc_type)
            db.update_step(request_id, "ocr", ocr_ms)
            db.update_ocr_meta(request_id, num_pages, total_blocks, doc_type)

            early_aadhaar = doc_type == "aadhaar"
            masking_key = None

            with ThreadPoolExecutor(max_workers=2) as pool:
                mask_future = None
                if early_aadhaar:
                    masking_key = f"{MASKING_PREFIX}/{filename}"
                    mask_future = pool.submit(
                        s3_client.copy_object,
                        CopySource={"Bucket": bucket, "Key": key},
                        Bucket=S3_BUCKET, Key=masking_key, MetadataDirective="COPY",
                    )
                    log.info("Aadhaar detected — early copy to s3://%s/%s", S3_BUCKET, masking_key)

                t0 = time.monotonic()
                llm_future = pool.submit(format_ocr_result, bedrock_client, ocr_result, BEDROCK_MODEL, doc_type)
                documents = llm_future.result()
                llm_ms = int((time.monotonic() - t0) * 1000)
                if mask_future is not None:
                    mask_future.result()

            log.info("[LATENCY] vlm_bedrock=%dms", llm_ms)
            db.update_step(request_id, "llm", llm_ms)

            has_aadhaar_in_json = any(d.get("documentType") == "AADHAAR" for d in documents)
            if early_aadhaar or has_aadhaar_in_json:
                for doc in documents:
                    if doc.get("documentType") == "AADHAAR":
                        doc.setdefault("additionalDetails", {})["outputMaskStatus"] = True

            output = {"requestId": request_id, "result": {"documents": documents}, "statusCode": 101}
            output_key = f"{OUTPUT_PREFIX}/{stem}.json"

            t0 = time.monotonic()
            with ThreadPoolExecutor(max_workers=2) as pool:
                save_future = pool.submit(
                    s3_client.put_object, Bucket=S3_BUCKET, Key=output_key,
                    Body=json.dumps(output, ensure_ascii=False, indent=4),
                    ContentType="application/json", ServerSideEncryption="AES256",
                )
                fallback_future = None
                if has_aadhaar_in_json and not early_aadhaar:
                    masking_key = f"{MASKING_PREFIX}/{filename}"
                    fallback_future = pool.submit(
                        s3_client.copy_object, CopySource={"Bucket": bucket, "Key": key},
                        Bucket=S3_BUCKET, Key=masking_key, MetadataDirective="COPY",
                    )
                    log.info("Aadhaar in JSON (regex missed) — fallback copy to s3://%s/%s", S3_BUCKET, masking_key)
                save_future.result()
                if fallback_future is not None:
                    fallback_future.result()
            save_ms = int((time.monotonic() - t0) * 1000)
            log.info("[LATENCY] json_save=%dms", save_ms)
            db.update_step(request_id, "jsonSave", save_ms)

            total_ms = int((time.monotonic() - pipeline_start) * 1000)
            db.complete_record(request_id, output_key, masking_key, documents, total_ms)
            doc_types = [d.get("documentType", "?") for d in documents]
            log.info(
                "[LATENCY] PIPELINE SUMMARY: download=%dms, engine_init=%dms, ocr=%dms, "
                "page_upload=%dms, classify=%dms, vlm=%dms, json_save=%dms, TOTAL=%dms | %d docs (%s)",
                download_ms, engine_ms, ocr_ms, page_upload_ms, classify_ms,
                llm_ms, save_ms, total_ms, len(documents), ", ".join(doc_types),
            )

            return {
                "statusCode": 200,
                "body": json.dumps({
                    "requestId": request_id, "input": key, "output": output_key,
                    "documents": len(documents), "document_types": doc_types,
                    "regex_doc_type": doc_type, "early_aadhaar": early_aadhaar,
                    "aadhaar_detected": has_aadhaar_in_json or early_aadhaar,
                    "masking_key": masking_key, "total_latency_ms": total_ms,
                }),
            }

        # ── PDF path (single pass at DPI 250) ─────────────────────────────────
        t0 = time.monotonic()
        ocr_result = process_pdf(engine, local_path)
        ocr_ms = int((time.monotonic() - t0) * 1000)
        num_pages = len(ocr_result.get("pages", []))
        total_blocks = ocr_result["total_blocks"]
        log.info("[LATENCY] ocr=%dms (%d blocks, %d pages)", ocr_ms, total_blocks, num_pages)

        t0 = time.monotonic()
        image_bytes_list = ocr_result.get("image_bytes_list", [])
        for idx, page_img_bytes in enumerate(image_bytes_list, 1):
            page_key = f"pipeline_pages/{stem}_p{idx}.jpg"
            try:
                s3_client.put_object(Bucket=S3_BUCKET, Key=page_key, Body=page_img_bytes, ContentType="image/jpeg")
            except Exception as exc:
                log.warning("Failed to save page image %d: %s", idx, exc)
        page_upload_ms = int((time.monotonic() - t0) * 1000)
        log.info("[LATENCY] page_images_upload=%dms (%d pages)", page_upload_ms, len(image_bytes_list))

        t0 = time.monotonic()
        doc_type = classify_document(ocr_result["full_text"])
        classify_ms = int((time.monotonic() - t0) * 1000)
        log.info("[LATENCY] classification=%dms → %s", classify_ms, doc_type)
        db.update_step(request_id, "ocr", ocr_ms)
        db.update_ocr_meta(request_id, num_pages, total_blocks, doc_type)

        early_aadhaar = doc_type == "aadhaar"
        masking_key = None

        with ThreadPoolExecutor(max_workers=2) as pool:
            mask_futures = []
            if early_aadhaar:
                for idx, page_img_bytes in enumerate(image_bytes_list, 1):
                    masking_img_key = f"{MASKING_PREFIX}/{stem}_p{idx}.jpg"
                    mask_futures.append(pool.submit(
                        s3_client.put_object,
                        Bucket=S3_BUCKET, Key=masking_img_key,
                        Body=page_img_bytes, ContentType="image/jpeg",
                        Metadata={"mask_digits": mask_digits},
                    ))
                    log.info("Aadhaar detected — page %d queued for s3://%s/%s masking",
                             idx, S3_BUCKET, masking_img_key)
                masking_key = f"{MASKING_PREFIX}/{stem}_p1.jpg"

            t0 = time.monotonic()
            llm_future = pool.submit(format_ocr_result, bedrock_client, ocr_result, BEDROCK_MODEL, doc_type)
            documents = llm_future.result()
            llm_ms = int((time.monotonic() - t0) * 1000)

            for f in mask_futures:
                try:
                    f.result()
                except Exception as exc:
                    log.warning("Masking upload failed: %s", exc)

        log.info("[LATENCY] vlm_bedrock=%dms", llm_ms)
        db.update_step(request_id, "llm", llm_ms)

        has_aadhaar_in_json = any(d.get("documentType") == "AADHAAR" for d in documents)
        if early_aadhaar or has_aadhaar_in_json:
            for doc in documents:
                if doc.get("documentType") == "AADHAAR":
                    doc.setdefault("additionalDetails", {})["outputMaskStatus"] = True

        output = {"requestId": request_id, "result": {"documents": documents}, "statusCode": 101}
        output_key = f"{OUTPUT_PREFIX}/{stem}.json"

        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as pool:
            save_future = pool.submit(
                s3_client.put_object, Bucket=S3_BUCKET, Key=output_key,
                Body=json.dumps(output, ensure_ascii=False, indent=4),
                ContentType="application/json", ServerSideEncryption="AES256",
            )
            fallback_futures = []
            if has_aadhaar_in_json and not early_aadhaar:
                for idx, page_img_bytes in enumerate(image_bytes_list, 1):
                    masking_img_key = f"{MASKING_PREFIX}/{stem}_p{idx}.jpg"
                    fallback_futures.append(pool.submit(
                        s3_client.put_object,
                        Bucket=S3_BUCKET, Key=masking_img_key,
                        Body=page_img_bytes, ContentType="image/jpeg",
                        Metadata={"mask_digits": mask_digits},
                    ))
                    log.info("Aadhaar in JSON (regex missed) — fallback page %d queued for s3://%s/%s",
                             idx, S3_BUCKET, masking_img_key)
                masking_key = f"{MASKING_PREFIX}/{stem}_p1.jpg"
            save_future.result()
            for f in fallback_futures:
                try:
                    f.result()
                except Exception as exc:
                    log.warning("Fallback masking upload failed: %s", exc)
        save_ms = int((time.monotonic() - t0) * 1000)
        log.info("[LATENCY] json_save=%dms", save_ms)
        db.update_step(request_id, "jsonSave", save_ms)

        total_ms = int((time.monotonic() - pipeline_start) * 1000)
        db.complete_record(request_id, output_key, masking_key, documents, total_ms)
        doc_types = [d.get("documentType", "?") for d in documents]
        log.info(
            "[LATENCY] PIPELINE SUMMARY: download=%dms, engine_init=%dms, ocr=%dms, "
            "page_upload=%dms, classify=%dms, vlm=%dms, json_save=%dms, TOTAL=%dms | %d docs (%s)",
            download_ms, engine_ms, ocr_ms, page_upload_ms, classify_ms,
            llm_ms, save_ms, total_ms, len(documents), ", ".join(doc_types),
        )

        return {
            "statusCode": 200,
            "body": json.dumps({
                "requestId": request_id, "input": key, "output": output_key,
                "documents": len(documents), "document_types": doc_types,
                "regex_doc_type": doc_type, "early_aadhaar": early_aadhaar,
                "aadhaar_detected": has_aadhaar_in_json or early_aadhaar,
                "masking_key": masking_key, "total_latency_ms": total_ms,
            }),
        }

    except Exception as exc:
        total_ms = int((time.monotonic() - pipeline_start) * 1000)
        log.error("Pipeline failed: %s", exc, exc_info=True)
        db.fail_record(request_id, str(exc), total_ms)
        raise

    finally:
        if local_path.exists():
            local_path.unlink()
