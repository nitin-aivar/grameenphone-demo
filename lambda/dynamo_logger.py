"""
DynamoDB logger for the OCR pipeline.

Writes one record per file processed with all metadata:
- S3 keys (input, output, masking, masked result)
- File type, extension, page count, OCR block count
- Document types and subtypes extracted
- Per-step status and latency (download, ocr, llm, save, masking)
- Total latency, timestamps, error messages
- TTL for auto-expiry after 90 days
"""
from __future__ import annotations

import logging
import os
import time
from decimal import Decimal
from typing import Any

import boto3
from botocore.exceptions import ClientError

log = logging.getLogger(__name__)

DYNAMO_TABLE = os.environ.get("DYNAMO_TABLE", "chola-ocr-pipeline-logs")

_dynamo_client = None


def _get_client():
    global _dynamo_client
    if _dynamo_client is None:
        _dynamo_client = boto3.resource("dynamodb", region_name=os.environ.get("AWS_REGION", "ap-south-1"))
    return _dynamo_client


def _table():
    return _get_client().Table(DYNAMO_TABLE)


def _to_decimal(val: float | int) -> Decimal:
    """Convert float to Decimal for DynamoDB (no float support)."""
    return Decimal(str(round(val, 3)))


def create_record(request_id: str, input_key: str, filename: str, file_type: str, file_ext: str) -> None:
    """Write the initial record when processing starts."""
    now = int(time.time())
    ttl = now + (90 * 24 * 3600)  # 90-day auto-expiry

    item = {
        "requestId": request_id,
        "inputKey": input_key,
        "filename": filename,
        "fileType": file_type,          # "image" | "pdf"
        "fileExtension": file_ext,      # ".png", ".pdf", etc.
        "status": "PROCESSING",
        "steps": {
            "download": {"completed": False, "latencyMs": 0},
            "ocr":      {"completed": False, "latencyMs": 0},
            "llm":      {"completed": False, "latencyMs": 0},
            "jsonSave": {"completed": False, "latencyMs": 0},
            "masking":  {"completed": False, "triggered": False},
        },
        "numPages": 0,
        "totalOcrBlocks": 0,
        "docTypeRegex": "",
        "numDocuments": 0,
        "documentSummary": [],          # [{documentType, subType}]
        "outputKey": "",
        "maskingInputKey": "",
        "totalLatencyMs": 0,
        "createdAt": now,
        "completedAt": 0,
        "errorMessage": "",
        "ttl": ttl,
    }

    try:
        _table().put_item(Item=item)
    except ClientError as exc:
        log.warning("DynamoDB create_record failed: %s", exc)


def update_step(request_id: str, step: str, latency_ms: int, **extra_fields) -> None:
    """Mark a pipeline step as completed with its latency."""
    update_expr = "SET steps.#s.completed = :t, steps.#s.latencyMs = :ms"
    expr_names = {"#s": step}
    expr_values: dict[str, Any] = {":t": True, ":ms": latency_ms}

    for k, v in extra_fields.items():
        placeholder = f":extra_{k}"
        update_expr += f", {k} = {placeholder}"
        if isinstance(v, float):
            v = _to_decimal(v)
        expr_values[placeholder] = v

    try:
        _table().update_item(
            Key={"requestId": request_id},
            UpdateExpression=update_expr,
            ExpressionAttributeNames=expr_names,
            ExpressionAttributeValues=expr_values,
        )
    except ClientError as exc:
        log.warning("DynamoDB update_step(%s, %s) failed: %s", request_id, step, exc)


def update_ocr_meta(request_id: str, num_pages: int, total_blocks: int, doc_type_regex: str) -> None:
    """Store OCR result metadata."""
    try:
        _table().update_item(
            Key={"requestId": request_id},
            UpdateExpression="SET numPages = :p, totalOcrBlocks = :b, docTypeRegex = :d",
            ExpressionAttributeValues={
                ":p": num_pages,
                ":b": total_blocks,
                ":d": doc_type_regex,
            },
        )
    except ClientError as exc:
        log.warning("DynamoDB update_ocr_meta failed: %s", exc)


def complete_record(
    request_id: str,
    output_key: str,
    masking_input_key: str,
    documents: list[dict],
    total_latency_ms: int,
) -> None:
    """Mark the record as COMPLETED with final output metadata."""
    doc_summary = [
        {"documentType": d.get("documentType", ""), "subType": d.get("subType", "")}
        for d in documents
    ]

    try:
        _table().update_item(
            Key={"requestId": request_id},
            UpdateExpression=(
                "SET #st = :s, outputKey = :ok, maskingInputKey = :mk, "
                "numDocuments = :nd, documentSummary = :ds, "
                "totalLatencyMs = :tl, completedAt = :ca, "
                "steps.masking.triggered = :mt"
            ),
            ExpressionAttributeNames={"#st": "status"},
            ExpressionAttributeValues={
                ":s": "COMPLETED",
                ":ok": output_key,
                ":mk": masking_input_key or "",
                ":nd": len(documents),
                ":ds": doc_summary,
                ":tl": total_latency_ms,
                ":ca": int(time.time()),
                ":mt": bool(masking_input_key),
            },
        )
    except ClientError as exc:
        log.warning("DynamoDB complete_record failed: %s", exc)


def fail_record(request_id: str, error_message: str, total_latency_ms: int) -> None:
    """Mark the record as FAILED with the error message."""
    try:
        _table().update_item(
            Key={"requestId": request_id},
            UpdateExpression=(
                "SET #st = :s, errorMessage = :e, "
                "totalLatencyMs = :tl, completedAt = :ca"
            ),
            ExpressionAttributeNames={"#st": "status"},
            ExpressionAttributeValues={
                ":s": "FAILED",
                ":e": str(error_message)[:1000],  # cap at 1000 chars
                ":tl": total_latency_ms,
                ":ca": int(time.time()),
            },
        )
    except ClientError as exc:
        log.warning("DynamoDB fail_record failed: %s", exc)
