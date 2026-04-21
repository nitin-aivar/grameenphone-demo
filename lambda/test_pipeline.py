#!/usr/bin/env python3
"""
Comprehensive test suite for the Chola OCR Lambda pipeline.

Tests every component end-to-end:
  1. doc_classifier  — regex classification per document type
  2. ocr_engine      — OCR processing, image bytes, PDF handling
  3. json_formatter  — prompt loading, JSON extraction, Verhoeff, enrichment
  4. Format validation — output JSON matches Chola/Karza ground truth schema
  5. E2E pipeline    — full OCR → classify → Bedrock → JSON for each doc type

Usage:
  cd lambda
  source .venv/bin/activate

  # Unit + format tests (no AWS credentials needed):
  python test_pipeline.py --unit

  # E2E tests with Bedrock (requires AWS credentials):
  python test_pipeline.py --e2e --profile my-sso-profile

  # Run everything:
  python test_pipeline.py --all --profile my-sso-profile

  # Single doc type E2E:
  python test_pipeline.py --e2e --profile my-sso-profile --doc-type aadhaar
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_pipeline")

# ── Test result tracking ──────────────────────────────────────────────────────

_results: list[dict] = []


def _record(name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    _results.append({"name": name, "passed": passed, "detail": detail})
    icon = "✓" if passed else "✗"
    msg = f"  {icon} {name}"
    if detail and not passed:
        msg += f"  — {detail}"
    log.info(msg)


def _section(title: str):
    log.info("")
    log.info("=" * 70)
    log.info("  %s", title)
    log.info("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
#  1. DOC CLASSIFIER TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_classifier():
    _section("1. Document Classifier (doc_classifier.py)")

    from doc_classifier import classify_document, is_aadhaar

    # --- Aadhaar ---
    aadhaar_text = """
    भारत सरकार GOVERNMENT OF INDIA
    Unique Identification Authority of India
    AADHAAR
    2637 4125 5075
    DOB: 24/07/2003
    VID: 9175946239821397
    """
    result = classify_document(aadhaar_text)
    _record("Classifier: Aadhaar text → 'aadhaar'", result == "aadhaar", f"got '{result}'")
    _record("is_aadhaar() returns True for Aadhaar text", is_aadhaar(aadhaar_text))

    # --- PAN ---
    pan_text = """
    INCOME TAX DEPARTMENT
    GOVT. OF INDIA
    PERMANENT ACCOUNT NUMBER
    BXKPG6694L
    Name: SANJAY GHOSH
    Father: BIPAD TARAN GHOSH
    DOB: 02/01/1974
    """
    result = classify_document(pan_text)
    _record("Classifier: PAN text → 'pan'", result == "pan", f"got '{result}'")
    _record("is_aadhaar() returns False for PAN text", not is_aadhaar(pan_text))

    # --- Passport ---
    passport_text = """
    REPUBLIC OF INDIA
    PASSPORT
    P<INDARUMUGAM<<SELVA<GANESH<<<<<<<<<<<<<<<<
    W9443344<5IND8910130M33020713074998377723<74
    Surname: ARUMUGAM
    Given Name: SELVA GANESH
    """
    result = classify_document(passport_text)
    _record("Classifier: Passport text → 'passport'", result == "passport", f"got '{result}'")

    # --- Voter ---
    voter_text = """
    ELECTION COMMISSION OF INDIA
    EPIC NO: WIC7896681
    Elector's Name: Chandrakant Yadav
    Father's Name: Mahadev Yadav
    """
    result = classify_document(voter_text)
    _record("Classifier: Voter text → 'voter'", result == "voter", f"got '{result}'")

    # --- E-Voter ---
    e_voter_text = """
    ELECTION COMMISSION OF INDIA
    e-EPIC
    Digital Voter ID
    EPIC NO: RTO4673000
    Name: Selva Ganesh A
    """
    result = classify_document(e_voter_text)
    _record("Classifier: E-Voter text → 'e_voter'", result == "e_voter", f"got '{result}'")

    # --- Generic fallback ---
    generic_text = "Random document text with no identifying markers 12345"
    result = classify_document(generic_text)
    _record("Classifier: Unknown text → 'generic'", result == "generic", f"got '{result}'")

    # --- Threshold test: weak signals should NOT classify ---
    weak_text = "Government of India"  # score=2 for aadhaar, below threshold=8
    result = classify_document(weak_text)
    _record("Classifier: Weak signal → 'generic' (below threshold)", result == "generic", f"got '{result}'")


# ══════════════════════════════════════════════════════════════════════════════
#  2. OCR ENGINE TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_ocr_engine(engine):
    _section("2. OCR Engine (ocr_engine.py)")

    from ocr_engine import process_image, IMAGE_EXTS, PDF_EXTS

    input_dir = Path("../code/input")
    test_image = input_dir / "Aadhar1.png"

    if not test_image.exists():
        _record("OCR Engine: test image exists", False, f"{test_image} not found")
        return

    # --- Process image ---
    t0 = time.time()
    result = process_image(engine, test_image)
    elapsed = time.time() - t0

    _record("OCR: returns dict", isinstance(result, dict))
    _record("OCR: has 'pages' key", "pages" in result)
    _record("OCR: has 'full_text' key", "full_text" in result)
    _record("OCR: has 'total_blocks' key", "total_blocks" in result)
    _record("OCR: has 'image_bytes_list' key", "image_bytes_list" in result)
    _record("OCR: has 'elapsed_seconds' key", "elapsed_seconds" in result)

    # --- image_bytes_list validation ---
    ibl = result.get("image_bytes_list", [])
    _record("OCR: image_bytes_list is non-empty list", isinstance(ibl, list) and len(ibl) > 0)
    if ibl:
        _record("OCR: image_bytes_list[0] is bytes", isinstance(ibl[0], bytes))
        _record("OCR: image bytes > 0", len(ibl[0]) > 0)

    # --- Pages structure ---
    pages = result.get("pages", [])
    _record("OCR: pages is non-empty list", len(pages) > 0)
    if pages:
        p = pages[0]
        _record("OCR: page has 'page' key (int)", isinstance(p.get("page"), int))
        _record("OCR: page has 'blocks' key (list)", isinstance(p.get("blocks"), list))
        _record("OCR: page has 'text' key (str)", isinstance(p.get("text"), str))

    # --- Block structure ---
    if pages and pages[0].get("blocks"):
        b = pages[0]["blocks"][0]
        _record("OCR: block has 'text'", "text" in b)
        _record("OCR: block has 'confidence'", "confidence" in b)
        _record("OCR: block has 'bounding_box'", "bounding_box" in b)
        _record("OCR: confidence is float 0-1", isinstance(b["confidence"], float) and 0 <= b["confidence"] <= 1)

    # --- Blocks count ---
    _record("OCR: total_blocks > 0", result["total_blocks"] > 0)
    _record("OCR: full_text is non-empty", len(result["full_text"]) > 0)

    log.info("  (OCR completed in %.1fs, %d blocks)", elapsed, result["total_blocks"])

    # --- Supported extensions ---
    _record("OCR: IMAGE_EXTS includes .png", ".png" in IMAGE_EXTS)
    _record("OCR: IMAGE_EXTS includes .jpg", ".jpg" in IMAGE_EXTS)
    _record("OCR: IMAGE_EXTS includes .jpeg", ".jpeg" in IMAGE_EXTS)
    _record("OCR: PDF_EXTS includes .pdf", ".pdf" in PDF_EXTS)


# ══════════════════════════════════════════════════════════════════════════════
#  3. JSON FORMATTER TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_json_formatter():
    _section("3. JSON Formatter (json_formatter.py)")

    from json_formatter import (
        extract_json_from_text,
        verhoeff_checksum,
        enrich_documents,
        _load_prompts,
        _get_prompt,
        _detect_media_type,
    )

    # ── 3a. Prompt loading ────────────────────────────────────────────────
    log.info("  3a. Prompt loading")
    prompts = _load_prompts()
    expected_keys = {"generic", "aadhaar", "pan", "passport", "voter", "e_voter"}
    _record("Prompts: loaded successfully", prompts is not None)
    _record("Prompts: has all 6 doc type keys", expected_keys.issubset(set(prompts.keys())),
            f"missing: {expected_keys - set(prompts.keys())}")

    for key in expected_keys:
        cfg = prompts.get(key, {})
        _record(f"Prompts: '{key}' has system_prompt", "system_prompt" in cfg)
        _record(f"Prompts: '{key}' has user_prompt_template", "user_prompt_template" in cfg)
        tmpl = cfg.get("user_prompt_template", "")
        _record(f"Prompts: '{key}' template has {{ocr_text}} placeholder", "{ocr_text}" in tmpl)

    # Test _get_prompt returns correct types
    sys_p, user_t = _get_prompt("aadhaar")
    _record("Prompts: _get_prompt returns (str, str)", isinstance(sys_p, str) and isinstance(user_t, str))
    # Fallback to generic
    sys_p2, _ = _get_prompt("nonexistent_type")
    sys_g, _ = _get_prompt("generic")
    _record("Prompts: unknown doc_type falls back to generic", sys_p2 == sys_g)

    # ── 3b. JSON extraction ───────────────────────────────────────────────
    log.info("  3b. JSON extraction")

    # Clean JSON
    parsed = extract_json_from_text('[{"documentType": "PAN"}]')
    _record("JSON extract: clean array", isinstance(parsed, list) and len(parsed) == 1)

    # With markdown fences
    parsed = extract_json_from_text('```json\n[{"documentType": "PAN"}]\n```')
    _record("JSON extract: markdown fenced", isinstance(parsed, list) and len(parsed) == 1)

    # With <think> tags (reasoning models)
    parsed = extract_json_from_text('<think>reasoning here</think>\n[{"documentType": "PAN"}]')
    _record("JSON extract: with <think> tags", isinstance(parsed, list) and len(parsed) == 1)

    # Nested in text
    parsed = extract_json_from_text('Here is the result:\n{"documentType": "AADHAAR"}\nDone.')
    _record("JSON extract: embedded in text", isinstance(parsed, dict))

    # Object with documents key
    parsed = extract_json_from_text('{"documents": [{"documentType": "PAN"}]}')
    _record("JSON extract: object → unwrap not a list directly",
            isinstance(parsed, dict) and "documents" in parsed)

    # ── 3c. Verhoeff checksum ─────────────────────────────────────────────
    log.info("  3c. Verhoeff checksum")

    # Known valid Aadhaar numbers (from test outputs)
    _record("Verhoeff: 263741255075 is valid", verhoeff_checksum("263741255075"))
    # Known invalid
    _record("Verhoeff: 123456789012 is invalid", not verhoeff_checksum("123456789012"))
    # Edge cases
    # Empty string: Verhoeff loop doesn't execute, c stays 0 → returns True (vacuous truth)
    _record("Verhoeff: empty string → True (vacuous)", verhoeff_checksum(""))
    _record("Verhoeff: non-digit → False", not verhoeff_checksum("abcdefghijkl"))
    _record("Verhoeff: short number → handles gracefully", not verhoeff_checksum("1234"))

    # ── 3d. Document enrichment ───────────────────────────────────────────
    log.info("  3d. Document enrichment (enrich_documents)")

    # AADHAAR document enrichment
    aadhaar_doc = {
        "documentType": "AADHAAR",
        "subType": "FRONT_BOTTOM",
        "pageNo": 1,
        "ocrData": {
            "aadhaar": {"value": "263741255075"},
            "name": {"value": "Test User"},
            "dob": {"value": "01/01/2000"},
            "gender": {"value": "MALE"},
            "father": {"value": ""},
            "husband": {"value": ""},
            "mother": {"value": ""},
            "vid": {"value": ""},
            "yob": {"value": "2000"},
        },
        "additionalDetails": {},
    }

    enriched = enrich_documents([aadhaar_doc])
    doc = enriched[0]
    ad = doc.get("additionalDetails", {})

    _record("Enrich: adds inputMaskStatus", "inputMaskStatus" in ad)
    _record("Enrich: inputMaskStatus has isMasked", "isMasked" in ad.get("inputMaskStatus", {}))
    _record("Enrich: inputMaskStatus.isMasked is False", ad.get("inputMaskStatus", {}).get("isMasked") is False)
    _record("Enrich: inputMaskStatus.maskedBy is None", ad.get("inputMaskStatus", {}).get("maskedBy") is None)
    _record("Enrich: inputMaskStatus.confidence is None", ad.get("inputMaskStatus", {}).get("confidence") is None)
    _record("Enrich: adds verhoeffCheck", "verhoeffCheck" in ad)
    _record("Enrich: verhoeffCheck is True for valid Aadhaar", ad.get("verhoeffCheck") is True)
    _record("Enrich: adds outputMaskStatus", "outputMaskStatus" in ad)
    _record("Enrich: outputMaskStatus is False", ad.get("outputMaskStatus") is False)
    _record("Enrich: adds qr (None)", "qr" in ad and ad["qr"] is None)
    _record("Enrich: adds barcode (None)", "barcode" in ad and ad["barcode"] is None)
    _record("Enrich: adds documentLink (None)", "documentLink" in doc and doc["documentLink"] is None)

    # Key ordering check (Chola format)
    ad_keys = list(ad.keys())
    expected_order = ["inputMaskStatus", "verhoeffCheck", "outputMaskStatus", "qr", "barcode"]
    _record("Enrich: key ordering matches Chola format",
            ad_keys[:5] == expected_order,
            f"got {ad_keys[:5]}")

    # Enrich with addressSplit preservation
    aadhaar_top = {
        "documentType": "AADHAAR",
        "subType": "FRONT_TOP",
        "pageNo": 1,
        "ocrData": {
            "aadhaar": {"value": "263741255075"},
            "name": {"value": "Test"},
            "address": {"value": "Some Address"},
            "phone": {"value": ""},
            "pin": {"value": "121001"},
            "vid": {"value": ""},
        },
        "additionalDetails": {
            "addressSplit": {
                "building": "", "city": "Test", "district": "", "pin": "121001",
                "floor": "", "house": "", "locality": "", "state": "Test",
                "street": "", "complex": "", "landmark": "", "untagged": "",
            },
            "careOfDetails": {"relation": "S/O", "name": "Father"},
        },
    }
    enriched2 = enrich_documents([aadhaar_top])
    ad2 = enriched2[0]["additionalDetails"]
    _record("Enrich: preserves addressSplit", "addressSplit" in ad2)
    _record("Enrich: preserves careOfDetails", "careOfDetails" in ad2)
    _record("Enrich: addressSplit comes after barcode",
            list(ad2.keys()).index("addressSplit") > list(ad2.keys()).index("barcode"))

    # Enrich Aadhaar with spaces in number
    spaced_doc = {
        "documentType": "AADHAAR", "subType": "FRONT_BOTTOM", "pageNo": 1,
        "ocrData": {"aadhaar": {"value": "2637 4125 5075"}},
        "additionalDetails": {},
    }
    enriched3 = enrich_documents([spaced_doc])
    _record("Enrich: strips spaces from Aadhaar number",
            enriched3[0]["ocrData"]["aadhaar"]["value"] == "263741255075")

    # Non-Aadhaar doc should NOT be enriched
    pan_doc = {
        "documentType": "PAN", "subType": "", "pageNo": 1,
        "ocrData": {"pan": {"value": "BXKPG6694L"}},
        "additionalDetails": {},
    }
    enriched4 = enrich_documents([pan_doc])
    _record("Enrich: PAN doc NOT enriched (no inputMaskStatus)",
            "inputMaskStatus" not in enriched4[0].get("additionalDetails", {}))
    _record("Enrich: PAN doc has no documentLink",
            "documentLink" not in enriched4[0])

    # ── 3e. Media type detection ──────────────────────────────────────────
    log.info("  3e. Media type detection")
    _record("Media: PNG magic bytes", _detect_media_type(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100) == "image/png")
    _record("Media: JPEG magic bytes", _detect_media_type(b'\xff\xd8' + b'\x00' * 100) == "image/jpeg")
    _record("Media: Unknown → image/jpeg", _detect_media_type(b'\x00\x00\x00\x00') == "image/jpeg")


# ══════════════════════════════════════════════════════════════════════════════
#  4. OUTPUT FORMAT VALIDATION (against Chola/Karza ground truth)
# ══════════════════════════════════════════════════════════════════════════════

# --- Ground truth schemas extracted from chola/*.json ---

AADHAAR_FRONT_TOP_FIELDS = {"aadhaar", "address", "name", "phone", "pin", "vid"}
AADHAAR_FRONT_BOTTOM_FIELDS = {"aadhaar", "dob", "father", "gender", "husband", "mother", "name", "vid", "yob"}
AADHAAR_BACK_FIELDS = {"aadhaar", "address", "pin", "vid"}

PAN_FIELDS = {"dob", "doi", "father", "name", "pan"}

PASSPORT_FRONT_FIELDS = {
    "countryCode", "dob", "doe", "doi", "gender", "givenName",
    "nationality", "passportNumber", "placeOfBirth", "placeOfIssue",
    "surname", "type", "mrzLine1", "mrzLine2",
}
PASSPORT_BACK_FIELDS = {
    "address", "father", "fileNumber", "mother", "oldDoi",
    "oldPassportNumber", "oldPlaceOfIssue", "passportNumber", "pin", "spouse",
}

VOTER_FRONT_FIELDS = {"voterId", "name", "dob", "gender", "ageAsPerDate", "relationName", "age"}
VOTER_BACK_FIELDS = {"voterId", "address", "lastUpdateDate"}

E_VOTER_FRONT_FIELDS = {"voterId", "name", "relationName", "gender", "dob", "age"}
E_VOTER_BACK_FIELDS = {"voterId", "address", "dob", "age", "gender", "pin"}

ADDRESS_SPLIT_KEYS = {
    "building", "city", "district", "pin", "floor", "house",
    "locality", "state", "street", "complex", "landmark", "untagged",
}

# Map doc_type + subType → expected ocrData fields
FIELD_SCHEMAS = {
    ("AADHAAR", "FRONT_TOP"): AADHAAR_FRONT_TOP_FIELDS,
    ("AADHAAR", "FRONT_BOTTOM"): AADHAAR_FRONT_BOTTOM_FIELDS,
    ("AADHAAR", "BACK"): AADHAAR_BACK_FIELDS,
    ("PAN", ""): PAN_FIELDS,
    ("PASSPORT", "FRONT"): PASSPORT_FRONT_FIELDS,
    ("PASSPORT", "BACK"): PASSPORT_BACK_FIELDS,
    ("VOTER", "FRONT"): VOTER_FRONT_FIELDS,
    ("VOTER", "BACK"): VOTER_BACK_FIELDS,
    ("E_VOTER", "FRONT"): E_VOTER_FRONT_FIELDS,
    ("E_VOTER", "BACK"): E_VOTER_BACK_FIELDS,
}


def validate_value_wrapper(ocr_data: dict, doc_label: str) -> list[str]:
    """Check that all ocrData values are wrapped in {"value": "..."} format."""
    errors = []
    for field_name, field_val in ocr_data.items():
        if not isinstance(field_val, dict):
            errors.append(f"{doc_label}.ocrData.{field_name}: expected dict, got {type(field_val).__name__}")
        elif "value" not in field_val:
            errors.append(f"{doc_label}.ocrData.{field_name}: missing 'value' key")
        elif not isinstance(field_val["value"], str):
            errors.append(f"{doc_label}.ocrData.{field_name}.value: expected str, got {type(field_val['value']).__name__}")
    return errors


def validate_address_split(addr_split: dict, doc_label: str) -> list[str]:
    """Validate addressSplit has exactly 12 plain string keys."""
    errors = []
    if set(addr_split.keys()) != ADDRESS_SPLIT_KEYS:
        missing = ADDRESS_SPLIT_KEYS - set(addr_split.keys())
        extra = set(addr_split.keys()) - ADDRESS_SPLIT_KEYS
        if missing:
            errors.append(f"{doc_label}.addressSplit: missing keys: {missing}")
        if extra:
            errors.append(f"{doc_label}.addressSplit: extra keys: {extra}")
    for k, v in addr_split.items():
        if not isinstance(v, str):
            errors.append(f"{doc_label}.addressSplit.{k}: expected plain str, got {type(v).__name__}")
    return errors


def validate_care_of_details(cod: dict, doc_label: str) -> list[str]:
    """Validate careOfDetails has plain string values."""
    errors = []
    expected_keys = {"relation", "name"}
    if set(cod.keys()) != expected_keys:
        errors.append(f"{doc_label}.careOfDetails: expected keys {expected_keys}, got {set(cod.keys())}")
    for k, v in cod.items():
        if not isinstance(v, str):
            errors.append(f"{doc_label}.careOfDetails.{k}: expected plain str, got {type(v).__name__}")
    return errors


def validate_document(doc: dict, doc_idx: int) -> list[str]:
    """Validate a single document dict against Chola/Karza format rules."""
    errors = []
    label = f"doc[{doc_idx}]"

    # Required top-level keys
    for key in ("documentType", "subType", "pageNo", "ocrData", "additionalDetails"):
        if key not in doc:
            errors.append(f"{label}: missing required key '{key}'")

    doc_type = doc.get("documentType", "")
    sub_type = doc.get("subType", "")
    page_no = doc.get("pageNo")
    ocr_data = doc.get("ocrData", {})
    additional = doc.get("additionalDetails", {})

    # documentType validation
    valid_types = {"AADHAAR", "PAN", "PASSPORT", "VOTER", "E_VOTER"}
    if doc_type not in valid_types:
        errors.append(f"{label}: documentType '{doc_type}' not in {valid_types}")

    # pageNo validation
    if not isinstance(page_no, int) or page_no < 1:
        errors.append(f"{label}: pageNo must be positive int, got {page_no}")

    # ocrData: all values must be {"value": "..."}
    errors.extend(validate_value_wrapper(ocr_data, label))

    # Field schema validation
    schema_key = (doc_type, sub_type)
    if schema_key in FIELD_SCHEMAS:
        expected_fields = FIELD_SCHEMAS[schema_key]
        actual_fields = set(ocr_data.keys())
        missing_fields = expected_fields - actual_fields
        if missing_fields:
            errors.append(f"{label} ({doc_type}/{sub_type}): missing ocrData fields: {missing_fields}")

    # AADHAAR-specific validations
    if doc_type == "AADHAAR":
        # Must have enrichment fields
        for aadhaar_key in ("inputMaskStatus", "verhoeffCheck", "outputMaskStatus", "qr", "barcode"):
            if aadhaar_key not in additional:
                errors.append(f"{label}: AADHAAR missing additionalDetails.{aadhaar_key}")

        # inputMaskStatus structure
        ims = additional.get("inputMaskStatus", {})
        if isinstance(ims, dict):
            for ims_key in ("isMasked", "maskedBy", "confidence"):
                if ims_key not in ims:
                    errors.append(f"{label}: inputMaskStatus missing '{ims_key}'")

        # documentLink must exist
        if "documentLink" not in doc:
            errors.append(f"{label}: AADHAAR missing 'documentLink' field")

    # addressSplit validation (when present)
    if "addressSplit" in additional:
        errors.extend(validate_address_split(additional["addressSplit"], label))

    # careOfDetails validation (when present)
    if "careOfDetails" in additional:
        errors.extend(validate_care_of_details(additional["careOfDetails"], label))

    # E_VOTER-specific: gender should be title case (Male/Female)
    if doc_type == "E_VOTER" and sub_type == "FRONT":
        gender_val = ocr_data.get("gender", {}).get("value", "")
        if gender_val and gender_val not in ("Male", "Female", ""):
            errors.append(f"{label}: E_VOTER gender should be title case (Male/Female), got '{gender_val}'")

    # E_VOTER-specific: dates should use hyphens DD-MM-YYYY
    if doc_type == "E_VOTER":
        dob_val = ocr_data.get("dob", {}).get("value", "")
        if dob_val and "/" in dob_val:
            errors.append(f"{label}: E_VOTER dates should use hyphens (DD-MM-YYYY), got '{dob_val}'")

    # Non-E_VOTER: dates should use slashes DD/MM/YYYY
    if doc_type not in ("E_VOTER",):
        dob_val = ocr_data.get("dob", {}).get("value", "")
        if dob_val and "-" in dob_val and dob_val.count("-") == 2:
            errors.append(f"{label}: {doc_type} dates should use slashes (DD/MM/YYYY), got '{dob_val}'")

    # PAN: subType should be "" (empty)
    if doc_type == "PAN" and sub_type != "":
        errors.append(f"{label}: PAN subType should be empty string, got '{sub_type}'")

    # PAN: additionalDetails should be empty {}
    if doc_type == "PAN" and additional != {}:
        errors.append(f"{label}: PAN additionalDetails should be {{}}, got keys: {list(additional.keys())}")

    # Gender UPPERCASE check (non-E_VOTER)
    if doc_type not in ("E_VOTER",):
        gender_val = ocr_data.get("gender", {}).get("value", "")
        if gender_val and gender_val not in ("MALE", "FEMALE", ""):
            errors.append(f"{label}: {doc_type} gender should be UPPERCASE, got '{gender_val}'")

    return errors


def validate_output_json(output: dict, source_label: str = "output"):
    """Validate a full pipeline output JSON against Chola format."""
    errors = []

    # Top-level structure
    for key in ("requestId", "result", "statusCode"):
        if key not in output:
            errors.append(f"Missing top-level key: '{key}'")

    if output.get("statusCode") != 101:
        errors.append(f"statusCode should be 101, got {output.get('statusCode')}")

    result = output.get("result", {})
    if "documents" not in result:
        errors.append("Missing result.documents")
        return errors

    documents = result["documents"]
    if not isinstance(documents, list) or len(documents) == 0:
        errors.append("result.documents should be a non-empty list")
        return errors

    for i, doc in enumerate(documents):
        errors.extend(validate_document(doc, i))

    return errors


def test_format_validation():
    _section("4. Output Format Validation (Chola/Karza schema)")

    # ── 4a. Validate test_output files ────────────────────────────────────
    log.info("  4a. Validating existing test_output files")

    test_output_dir = Path("./test_output")
    if not test_output_dir.exists():
        _record("Format: test_output/ directory exists", False, "not found")
        return

    json_files = sorted(test_output_dir.glob("*.json"))
    json_files = [f for f in json_files if not f.name.endswith("_ocr.json")]

    if not json_files:
        _record("Format: test_output has JSON files", False, "no non-OCR JSON files found")
        return

    for jf in json_files:
        output = json.loads(jf.read_text(encoding="utf-8"))
        errors = validate_output_json(output, jf.name)
        _record(
            f"Format: {jf.name} matches Chola schema",
            len(errors) == 0,
            "; ".join(errors[:3]) + (f" (+{len(errors)-3} more)" if len(errors) > 3 else "") if errors else "",
        )

    # ── 4b. Validate chola/ ground truth files ────────────────────────────
    log.info("  4b. Validating Chola ground truth files")

    chola_dir = Path("../chola")
    if not chola_dir.exists():
        _record("Format: chola/ directory exists", False, "not found")
        return

    chola_files = {
        "Aadhar (1).json": "AADHAAR",
        "PAN.json": "PAN",
        "Passport.json": "PASSPORT",
        "Voter-Id-Card.json": "VOTER",
        "E-Voter.json": "E_VOTER",
    }

    for filename, expected_type in chola_files.items():
        fpath = chola_dir / filename
        if not fpath.exists():
            _record(f"Format: chola/{filename} exists", False)
            continue

        output = json.loads(fpath.read_text(encoding="utf-8"))
        errors = validate_output_json(output, filename)
        _record(
            f"Format: chola/{filename} self-validates",
            len(errors) == 0,
            "; ".join(errors[:3]) if errors else "",
        )

        # Check document types match expected
        doc_types = [d["documentType"] for d in output["result"]["documents"]]
        all_match = all(dt == expected_type for dt in doc_types)
        _record(
            f"Format: chola/{filename} all docs are {expected_type}",
            all_match,
            f"got {doc_types}" if not all_match else "",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  5. HANDLER UNIT TESTS (no AWS calls)
# ══════════════════════════════════════════════════════════════════════════════

def test_handler_logic():
    _section("5. Handler Logic (handler.py)")

    from handler import _PIIRedactFilter

    # --- PII Redaction Filter ---
    log.info("  5a. PII redaction filter")

    filt = _PIIRedactFilter()

    class FakeRecord:
        def __init__(self, msg, args=None):
            self.msg = msg
            self.args = args
        def getMessage(self):
            if self.args:
                return self.msg % self.args
            return self.msg

    # Aadhaar number redaction (spaced)
    rec = FakeRecord("Processing 2637 4125 5075")
    filt.filter(rec)
    _record("PII: Aadhaar (spaced) is redacted", "2637" not in rec.msg and "XXXX" in rec.msg,
            f"got: {rec.msg}")

    # 12-digit continuous — the spaced pattern (\d{4}\s?\d{4}\s?\d{4}) matches first,
    # so continuous numbers get redacted as "XXXX XXXX 5075" (not "XXXXXXXX5075")
    rec = FakeRecord("Number: 263741255075")
    filt.filter(rec)
    _record("PII: Aadhaar (continuous) is redacted", "26374125" not in rec.msg,
            f"got: {rec.msg}")

    # VID redaction
    rec = FakeRecord("VID: 9175946239821397")
    filt.filter(rec)
    _record("PII: VID is redacted", "9175946239821397" not in rec.msg,
            f"got: {rec.msg}")

    # Non-PII should not be redacted
    rec = FakeRecord("Processing PAN: BXKPG6694L")
    filt.filter(rec)
    _record("PII: PAN number NOT redacted", "BXKPG6694L" in rec.msg)

    # Args formatting
    rec = FakeRecord("s3://%s/%s", ("my-bucket", "pipeline_input/file.png"))
    filt.filter(rec)
    _record("PII: args are formatted into msg", "my-bucket" in rec.msg)
    _record("PII: args cleared after formatting", rec.args is None)


# ══════════════════════════════════════════════════════════════════════════════
#  6. E2E PIPELINE TESTS (requires AWS credentials + Bedrock)
# ══════════════════════════════════════════════════════════════════════════════

def test_e2e_pipeline(engine, bedrock_client, model_id: str, doc_types_filter: str | None = None):
    _section("6. E2E Pipeline (OCR → Classify → Bedrock → Validate)")

    from ocr_engine import process_image
    from doc_classifier import classify_document
    from json_formatter import format_ocr_result

    input_dir = Path("../code/input")

    # Test cases: (filename, expected_regex_type, expected_doc_type_in_json, expected_subtypes)
    test_cases = [
        ("Aadhar1.png", "aadhaar", "AADHAAR", {"FRONT_BOTTOM", "FRONT_TOP"}),
        ("PAN1.png", "pan", "PAN", {""}),
        ("Passport1.png", "passport", "PASSPORT", {"FRONT"}),
        ("Voter1.png", "voter", "VOTER", {"FRONT"}),
    ]

    if doc_types_filter:
        allowed = {d.strip().lower() for d in doc_types_filter.split(",")}
        test_cases = [tc for tc in test_cases if tc[1] in allowed]
        log.info("  Filtered to doc types: %s", allowed)

    for filename, expected_regex, expected_json_type, expected_subtypes in test_cases:
        image_path = input_dir / filename
        if not image_path.exists():
            _record(f"E2E [{filename}]: file exists", False, f"{image_path} not found")
            continue

        log.info("")
        log.info("  ── E2E: %s ──", filename)

        try:
            # Step 1: OCR
            t0 = time.time()
            ocr_result = process_image(engine, image_path)
            ocr_time = time.time() - t0
            _record(f"E2E [{filename}]: OCR success ({ocr_result['total_blocks']} blocks, {ocr_time:.1f}s)", True)

            # Step 2: Classify
            doc_type = classify_document(ocr_result["full_text"])
            _record(f"E2E [{filename}]: regex classifies as '{expected_regex}'",
                    doc_type == expected_regex, f"got '{doc_type}'")

            # Step 3: Bedrock call
            t1 = time.time()
            documents = format_ocr_result(bedrock_client, ocr_result, model_id, doc_type)
            llm_time = time.time() - t1
            _record(f"E2E [{filename}]: Bedrock returns documents ({llm_time:.1f}s)",
                    isinstance(documents, list) and len(documents) > 0,
                    f"got {len(documents)} docs")

            # Step 4: Validate document types in output
            json_types = {d.get("documentType") for d in documents}
            _record(f"E2E [{filename}]: all docs are {expected_json_type}",
                    all(t == expected_json_type for t in json_types),
                    f"got types: {json_types}")

            # Step 5: Validate subtypes
            actual_subtypes = {d.get("subType", "") for d in documents}
            has_expected = expected_subtypes.issubset(actual_subtypes)
            _record(f"E2E [{filename}]: has expected subTypes {expected_subtypes}",
                    has_expected, f"got {actual_subtypes}")

            # Step 6: Build full output and validate format
            import uuid
            output = {
                "requestId": str(uuid.uuid4()),
                "result": {"documents": documents},
                "statusCode": 101,
            }
            errors = validate_output_json(output, filename)
            _record(f"E2E [{filename}]: output matches Chola schema ({len(errors)} errors)",
                    len(errors) == 0,
                    "; ".join(errors[:5]) if errors else "")

            # Step 7: Field completeness — check key fields are non-empty
            for doc in documents:
                dt = doc["documentType"]
                st = doc.get("subType", "")
                ocr = doc.get("ocrData", {})
                label = f"{dt}/{st}"

                if dt == "AADHAAR":
                    aadhaar_val = ocr.get("aadhaar", {}).get("value", "")
                    _record(f"E2E [{filename}]: {label} aadhaar is non-empty",
                            len(aadhaar_val) > 0, f"got '{aadhaar_val}'")
                    name_val = ocr.get("name", {}).get("value", "")
                    _record(f"E2E [{filename}]: {label} name is non-empty",
                            len(name_val) > 0, f"got '{name_val}'")

                elif dt == "PAN":
                    pan_val = ocr.get("pan", {}).get("value", "")
                    _record(f"E2E [{filename}]: PAN number is non-empty",
                            len(pan_val) > 0, f"got '{pan_val}'")
                    name_val = ocr.get("name", {}).get("value", "")
                    _record(f"E2E [{filename}]: PAN name is non-empty",
                            len(name_val) > 0, f"got '{name_val}'")

                elif dt == "PASSPORT" and st == "FRONT":
                    pp_val = ocr.get("passportNumber", {}).get("value", "")
                    _record(f"E2E [{filename}]: passport number is non-empty",
                            len(pp_val) > 0, f"got '{pp_val}'")

                elif dt == "VOTER" and st == "FRONT":
                    vid = ocr.get("voterId", {}).get("value", "")
                    _record(f"E2E [{filename}]: voterId is non-empty",
                            len(vid) > 0, f"got '{vid}'")

            # Save E2E result
            e2e_output_dir = Path("./test_output/e2e")
            e2e_output_dir.mkdir(parents=True, exist_ok=True)
            out_path = e2e_output_dir / f"{image_path.stem}_e2e.json"
            out_path.write_text(json.dumps(output, ensure_ascii=False, indent=4), encoding="utf-8")
            log.info("  Saved → %s", out_path)

        except Exception as exc:
            _record(f"E2E [{filename}]: pipeline completed", False, f"{type(exc).__name__}: {exc}")
            traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
#  7. CROSS-VALIDATION: test_output vs Chola ground truth
# ══════════════════════════════════════════════════════════════════════════════

def test_cross_validation():
    _section("7. Cross-Validation (test_output vs Chola schema rules)")

    test_output_dir = Path("./test_output")
    if not test_output_dir.exists():
        _record("Cross-validation: test_output/ exists", False)
        return

    json_files = sorted(f for f in test_output_dir.glob("*.json") if not f.name.endswith("_ocr.json"))

    for jf in json_files:
        output = json.loads(jf.read_text(encoding="utf-8"))
        documents = output.get("result", {}).get("documents", [])

        for i, doc in enumerate(documents):
            dt = doc.get("documentType", "")
            st = doc.get("subType", "")
            label = f"{jf.name}:doc[{i}]({dt}/{st})"

            # Null check — no null values in ocrData
            ocr = doc.get("ocrData", {})
            has_null = any(
                v.get("value") is None
                for v in ocr.values()
                if isinstance(v, dict)
            )
            _record(f"CrossVal: {label} — no null values in ocrData", not has_null)

            # Empty string check — missing values should be "" not absent
            schema_key = (dt, st)
            if schema_key in FIELD_SCHEMAS:
                expected = FIELD_SCHEMAS[schema_key]
                actual = set(ocr.keys())
                missing = expected - actual
                _record(f"CrossVal: {label} — all expected fields present",
                        len(missing) == 0, f"missing: {missing}" if missing else "")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def build_engine_from_hf():
    """Download models from HuggingFace and build OCR engine (same as test_local.py)."""
    import yaml
    from huggingface_hub import hf_hub_download
    from rapidocr_onnxruntime import RapidOCR
    import rapidocr_onnxruntime

    log.info("Downloading models from HuggingFace ...")
    repo = "monkt/paddleocr-onnx"
    det = hf_hub_download(repo, "detection/v5/det.onnx")
    rec = hf_hub_download(repo, "languages/english/rec.onnx")
    dct = hf_hub_download(repo, "languages/english/dict.txt")

    pkg_dir = Path(rapidocr_onnxruntime.__file__).parent
    cfg_path = pkg_dir / "config.yaml"
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    config["Det"]["model_path"] = det
    config["Rec"]["model_path"] = rec
    config["Rec"]["rec_keys_path"] = dct

    cfg_out = "/tmp/rapidocr_config_test.yaml"
    with open(cfg_out, "w") as f:
        yaml.dump(config, f)

    engine = RapidOCR(config_path=cfg_out)
    log.info("OCR engine ready.")
    return engine


def print_summary():
    _section("TEST SUMMARY")
    total = len(_results)
    passed = sum(1 for r in _results if r["passed"])
    failed = total - passed

    log.info("  Total : %d", total)
    log.info("  Passed: %d", passed)
    log.info("  Failed: %d", failed)

    if failed > 0:
        log.info("")
        log.info("  FAILURES:")
        for r in _results:
            if not r["passed"]:
                detail = f" — {r['detail']}" if r['detail'] else ""
                log.info("    ✗ %s%s", r["name"], detail)

    log.info("")
    log.info("  Result: %s", "ALL PASSED" if failed == 0 else f"{failed} FAILED")
    log.info("=" * 70)

    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Lambda pipeline test suite")
    parser.add_argument("--unit", action="store_true", help="Run unit + format tests only (no AWS)")
    parser.add_argument("--e2e", action="store_true", help="Run E2E tests (requires AWS credentials)")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--profile", default=None, help="AWS CLI profile for Bedrock")
    parser.add_argument("--model", default="mistral.ministral-3-14b-instruct", help="Bedrock model ID")
    parser.add_argument("--region", default="ap-south-1", help="Bedrock region")
    parser.add_argument("--doc-type", default=None, help="Comma-separated doc types to test (e2e only)")
    args = parser.parse_args()

    if not args.unit and not args.e2e and not args.all:
        parser.error("Specify --unit, --e2e, or --all")

    run_unit = args.unit or args.all
    run_e2e = args.e2e or args.all

    engine = None
    bedrock_client = None

    # Build OCR engine if needed
    if run_unit or run_e2e:
        log.info("Building OCR engine from HuggingFace...")
        engine = build_engine_from_hf()

    # Set up Bedrock client if needed
    if run_e2e:
        import boto3
        session_kwargs = {}
        if args.profile:
            session_kwargs["profile_name"] = args.profile
        session = boto3.Session(**session_kwargs)
        bedrock_client = session.client("bedrock-runtime", region_name=args.region)

    # ── Run tests ──
    t_start = time.time()

    if run_unit:
        test_classifier()
        test_ocr_engine(engine)
        test_json_formatter()
        test_format_validation()
        test_handler_logic()
        test_cross_validation()

    if run_e2e:
        test_e2e_pipeline(engine, bedrock_client, args.model, args.doc_type)

    elapsed = time.time() - t_start
    log.info("")
    log.info("Total test time: %.1fs", elapsed)

    all_passed = print_summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
