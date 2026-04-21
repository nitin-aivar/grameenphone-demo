"""
JSON Formatter for Lambda — Mistral vision extraction + enrichment.

Sends document images + OCR text to Mistral Ministral 3 14B via Bedrock
invoke_model API. Prompts are loaded from prompts.yaml and selected based
on the regex-classified document type.
"""
from __future__ import annotations

import base64
import json
import re
import time
import logging
from io import BytesIO

from doc_classifier import classify_document
from pathlib import Path

import yaml
from botocore.exceptions import ClientError

log = logging.getLogger(__name__)

MAX_RETRIES = 3
_MAX_IMAGE_BYTES = 4_500_000  # 4.5 MB (Bedrock limit ~5 MB)

# ── Prompt Loading ──────────────────────────────────────────────────────────

_prompts: dict | None = None


def _load_prompts() -> dict:
    """Load prompts.yaml from the same directory, cached at module level."""
    global _prompts
    if _prompts is not None:
        return _prompts

    prompts_path = Path(__file__).parent / "prompts.yaml"
    with open(prompts_path, encoding="utf-8") as f:
        _prompts = yaml.safe_load(f)
    log.info("Loaded prompts from %s (%d doc types)", prompts_path, len(_prompts))
    return _prompts


def _get_prompt(doc_type: str) -> tuple[str, str]:
    """Return (system_prompt, user_prompt_template) for the given doc type."""
    prompts = _load_prompts()
    cfg = prompts.get(doc_type) or prompts.get("generic", {})
    system = cfg.get("system_prompt", "")
    user_template = cfg.get("user_prompt_template", "OCR Text:\n{ocr_text}")
    return system, user_template


# ── Image Helpers ───────────────────────────────────────────────────────────

def _detect_media_type(image_bytes: bytes) -> str:
    """Detect image media type from magic bytes."""
    if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    if image_bytes[:2] == b'\xff\xd8':
        return "image/jpeg"
    if image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
        return "image/webp"
    if image_bytes[:4] in (b'II\x2a\x00', b'MM\x00\x2a'):
        return "image/tiff"
    if image_bytes[:2] == b'BM':
        return "image/bmp"
    # Default to JPEG (safest for vision models)
    return "image/jpeg"


def _resize_image_if_needed(image_bytes: bytes) -> tuple[bytes, str]:
    """Resize image if it exceeds the Bedrock size limit. Returns (bytes, media_type)."""
    media_type = _detect_media_type(image_bytes)

    if len(image_bytes) <= _MAX_IMAGE_BYTES:
        return image_bytes, media_type

    from PIL import Image
    with Image.open(BytesIO(image_bytes)) as img:
        ratio = (_MAX_IMAGE_BYTES / len(image_bytes)) ** 0.5
        new_size = (max(1, int(img.width * ratio)), max(1, int(img.height * ratio)))
        resized = img.resize(new_size, Image.LANCZOS)
        buf = BytesIO()
        resized.convert("RGB").save(buf, format="JPEG", quality=85)
        data = buf.getvalue()
        log.info("Resized image to %dx%d (%d bytes)", new_size[0], new_size[1], len(data))
        return data, "image/jpeg"


# ── Verhoeff Algorithm (Aadhaar checksum validation) ────────────────────────

_VERHOEFF_D = [
    [0,1,2,3,4,5,6,7,8,9], [1,2,3,4,0,6,7,8,9,5],
    [2,3,4,0,1,7,8,9,5,6], [3,4,0,1,2,8,9,5,6,7],
    [4,0,1,2,3,9,5,6,7,8], [5,9,8,7,6,0,4,3,2,1],
    [6,5,9,8,7,1,0,4,3,2], [7,6,5,9,8,2,1,0,4,3],
    [8,7,6,5,9,3,2,1,0,4], [9,8,7,6,5,4,3,2,1,0],
]
_VERHOEFF_P = [
    [0,1,2,3,4,5,6,7,8,9], [1,5,7,6,2,8,3,0,9,4],
    [5,8,0,3,7,9,6,1,4,2], [8,9,1,6,0,4,3,5,2,7],
    [9,4,5,3,1,2,6,8,7,0], [4,2,8,6,5,7,3,9,0,1],
    [2,7,9,3,8,0,6,4,1,5], [7,0,4,6,9,1,3,2,5,8],
]


def verhoeff_checksum(number_str: str) -> bool:
    """Return True if the digit string passes the Verhoeff check."""
    try:
        c = 0
        for i, ch in enumerate(reversed(number_str)):
            c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
        return c == 0
    except (ValueError, IndexError):
        return False


# ── JSON Extraction ─────────────────────────────────────────────────────────

def extract_json_from_text(text: str):
    """Robustly extract a JSON array or object from model output."""
    text = text.strip()
    # Strip <think>...</think> tags (reasoning models)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
        text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find JSON substring — track both braces and brackets independently
    first_brace = text.find('{')
    first_bracket = text.find('[')
    if first_brace == -1 and first_bracket == -1:
        return json.loads(text)  # will raise
    if first_bracket == -1:
        idx = first_brace
    elif first_brace == -1:
        idx = first_bracket
    else:
        idx = min(first_brace, first_bracket)

    # Track brace/bracket stack
    stack = []
    in_string = False
    escape = False
    for i in range(idx, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ('{', '['):
            stack.append(ch)
        elif ch == '}':
            if stack and stack[-1] == '{':
                stack.pop()
            if not stack:
                candidate = text[idx:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
        elif ch == ']':
            if stack and stack[-1] == '[':
                stack.pop()
            if not stack:
                candidate = text[idx:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

    # Truncated output — try closing open braces/brackets
    if stack:
        truncated = text[idx:]
        for opener in reversed(stack):
            truncated += '}' if opener == '{' else ']'
        try:
            return json.loads(truncated)
        except json.JSONDecodeError:
            pass

    # Try repairing missing braces before final bracket
    if text.rstrip().endswith(']'):
        inner = text[idx:]
        brace_depth = 0
        bracket_depth = 0
        in_str = False
        esc = False
        for ch in inner:
            if esc:
                esc = False
                continue
            if ch == '\\' and in_str:
                esc = True
                continue
            if ch == '"' and not esc:
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == '{':
                brace_depth += 1
            elif ch == '}':
                brace_depth -= 1
            elif ch == '[':
                bracket_depth += 1
            elif ch == ']':
                bracket_depth -= 1
        if brace_depth > 0 and bracket_depth == 0:
            fixed = inner[:-1] + '}' * brace_depth + ']'
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

    # Final fallback — raise the original error
    return json.loads(text)


# ── Mistral Vision Interaction ──────────────────────────────────────────────

def call_mistral_vision(
    client,
    system_prompt: str,
    user_message: str,
    image_bytes: bytes | None,
    model_id: str,
) -> list[dict]:
    """Send image + OCR text to Mistral vision model via invoke_model API."""
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            # Build user content
            user_content = []
            if image_bytes is not None:
                img_data, media_type = _resize_image_if_needed(image_bytes)
                b64_data = base64.standard_b64encode(img_data).decode("utf-8")
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{b64_data}"},
                })
            user_content.append({"type": "text", "text": user_message})

            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_content})

            body = {
                "messages": messages,
                "max_tokens": 4096,
                "temperature": 0.0,
            }

            t_invoke = time.time()
            response = client.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            invoke_ms = int((time.time() - t_invoke) * 1000)

            result = json.loads(response["body"].read())
            result_text = result["choices"][0]["message"]["content"]

            usage = result.get("usage", {})
            in_tok = usage.get("prompt_tokens", 0)
            out_tok = usage.get("completion_tokens", 0)
            log.info(
                "[LATENCY] bedrock_invoke=%dms (model=%s, tokens_in=%d, tokens_out=%d)",
                invoke_ms, model_id, in_tok, out_tok,
            )

            parsed = extract_json_from_text(result_text)

            # Normalise: accept both bare array and {"documents": [...]}
            if isinstance(parsed, dict):
                parsed = parsed.get("documents", [parsed])
            if not isinstance(parsed, list):
                parsed = [parsed]

            return parsed

        except json.JSONDecodeError as exc:
            last_error = exc
            log.warning(
                "Invalid JSON on attempt %d/%d — retrying …",
                attempt + 1,
                MAX_RETRIES,
            )
            time.sleep(2**attempt)

        except ClientError as exc:
            error_code = exc.response["Error"]["Code"]
            if error_code in (
                "ThrottlingException",
                "ServiceUnavailableException",
                "ModelTimeoutException",
            ):
                last_error = exc
                wait = 2**attempt
                log.warning(
                    "%s on attempt %d/%d — waiting %ds …",
                    error_code,
                    attempt + 1,
                    MAX_RETRIES,
                    wait,
                )
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(
        f"Failed after {MAX_RETRIES} attempts: {last_error}"
    ) from last_error


# ── Post-processing ──────────────────────────────────────────────────────────

def _fix_aadhaar_group_order(number_str: str) -> str:
    """If a 12-digit Aadhaar number fails Verhoeff, try all 6 permutations of its
    three 4-digit groups. Physical card OCR sometimes reads groups in wrong order.
    Returns the first permutation that passes, or the original string if none does."""
    from itertools import permutations
    if len(number_str) != 12 or not number_str.isdigit():
        return number_str
    groups = [number_str[i:i + 4] for i in range(0, 12, 4)]
    for perm in permutations(groups):
        candidate = "".join(perm)
        if verhoeff_checksum(candidate):
            return candidate
    return number_str


def enrich_documents(documents: list[dict]) -> list[dict]:
    """Add computed fields (Verhoeff check, mask status, documentLink) to Aadhaar docs.

    Ensures additionalDetails key ordering matches Chola/Karza format:
    inputMaskStatus, verhoeffCheck, outputMaskStatus, qr, barcode, [addressSplit], [careOfDetails]
    """
    for doc in documents:
        doc_type = doc.get("documentType", "")
        sub_type = doc.get("subType", "UNKNOWN")

        if doc_type == "AADHAAR":
            aadhaar_val = (
                doc.get("ocrData", {}).get("aadhaar", {}).get("value", "")
            )
            # Clean up any spaces/hyphens in the extracted number
            clean_aadhaar = re.sub(r"[\s\-]", "", aadhaar_val)
            if clean_aadhaar != aadhaar_val and clean_aadhaar.isdigit():
                doc["ocrData"]["aadhaar"]["value"] = clean_aadhaar
                aadhaar_val = clean_aadhaar

            # If Verhoeff fails, OCR may have read the three 4-digit groups in the
            # wrong order (common with physical cards photographed at an angle).
            # Try all 6 permutations to recover the correct ordering.
            if (
                len(aadhaar_val) == 12
                and aadhaar_val.isdigit()
                and not verhoeff_checksum(aadhaar_val)
            ):
                corrected = _fix_aadhaar_group_order(aadhaar_val)
                if corrected != aadhaar_val:
                    log.info(
                        "Aadhaar group order corrected: %s -> %s",
                        aadhaar_val[-4:],  # log only last 4 for PII safety
                        corrected[-4:],
                    )
                    doc["ocrData"]["aadhaar"]["value"] = corrected
                    aadhaar_val = corrected

            is_valid = (
                len(aadhaar_val) == 12
                and aadhaar_val.isdigit()
                and verhoeff_checksum(aadhaar_val)
            )

            old_ad = doc.get("additionalDetails", {})

            # Rebuild additionalDetails with correct key ordering (Chola format)
            new_ad = {}
            new_ad["inputMaskStatus"] = {
                "isMasked": False,
                "maskedBy": None,
                "confidence": None,
            }
            new_ad["verhoeffCheck"] = is_valid
            new_ad["outputMaskStatus"] = False
            new_ad["qr"] = None
            new_ad["barcode"] = None

            # Address-related fields come AFTER aadhaar-specific fields
            if "addressSplit" in old_ad:
                new_ad["addressSplit"] = old_ad["addressSplit"]
            if "careOfDetails" in old_ad:
                new_ad["careOfDetails"] = old_ad["careOfDetails"]

            doc["additionalDetails"] = new_ad

            # Add documentLink field (null since we don't have Karza download URLs)
            doc["documentLink"] = None

    return documents


# ── Main Entry Point ─────────────────────────────────────────────────────────

def format_ocr_result(
    bedrock_client, ocr_result: dict, model_id: str, doc_type: str = "generic"
) -> list[dict]:
    """Take raw OCR result dict, call Mistral vision, enrich, and return document list.

    Args:
        bedrock_client: boto3 bedrock-runtime client.
        ocr_result: Dict from ocr_engine with pages, full_text, image_bytes_list.
        model_id: Bedrock model ID (e.g. "mistral.ministral-3-14b-instruct").
        doc_type: Pre-classified document type for prompt selection.
    """
    pages = ocr_result.get("pages", [])
    image_bytes_list = ocr_result.get("image_bytes_list", [])

    # Load the appropriate prompt
    system_prompt, user_template = _get_prompt(doc_type)

    all_documents = []

    if len(pages) > 1:
        for i, page in enumerate(pages):
            page_text = page.get("text", "")
            page_image = image_bytes_list[i] if i < len(image_bytes_list) else None

            page_doc_type = classify_document(page_text)
            page_system_prompt, page_user_template = _get_prompt(page_doc_type)
            user_msg = page_user_template.format(ocr_text=page_text)

            log.info(
                "Processing page %d/%d with Mistral vision (doc_type=%s)",
                i + 1, len(pages), page_doc_type,
            )
            t_page = time.time()
            docs = call_mistral_vision(
                bedrock_client, page_system_prompt, user_msg, page_image, model_id
            )
            page_vlm_ms = int((time.time() - t_page) * 1000)
            log.info("[LATENCY] vlm_page_%d=%dms (doc_type=%s)", i + 1, page_vlm_ms, page_doc_type)

            for doc in docs:
                doc["pageNo"] = page["page"]
            all_documents.extend(docs)
    else:
        ocr_text = ocr_result.get("full_text", "")
        page_image = image_bytes_list[0] if image_bytes_list else None
        user_msg = user_template.format(ocr_text=ocr_text)

        log.info("Processing single image with Mistral vision (doc_type=%s)", doc_type)
        t_single = time.time()
        all_documents = call_mistral_vision(
            bedrock_client, system_prompt, user_msg, page_image, model_id
        )
        single_vlm_ms = int((time.time() - t_single) * 1000)
        log.info("[LATENCY] vlm_single=%dms (doc_type=%s)", single_vlm_ms, doc_type)

    # Enrich with computed fields
    all_documents = enrich_documents(all_documents)

    # Reorder Aadhaar sections to match Karza format: FRONT_TOP → FRONT_BOTTOM → BACK
    _AADHAAR_SUBTYPE_ORDER = {"FRONT_TOP": 0, "FRONT_BOTTOM": 1, "BACK": 2}
    aadhaar_docs = [d for d in all_documents if d.get("documentType") == "AADHAAR"]
    other_docs = [d for d in all_documents if d.get("documentType") != "AADHAAR"]
    aadhaar_docs.sort(key=lambda d: _AADHAAR_SUBTYPE_ORDER.get(d.get("subType", ""), 99))
    all_documents = aadhaar_docs + other_docs

    return all_documents
