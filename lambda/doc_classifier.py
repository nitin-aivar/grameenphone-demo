"""Lightweight document classifier using weighted regex pattern matching.

Classifies OCR text into one of: aadhaar, pan, passport, voter, e_voter, or
generic.  Each document type has a set of regex patterns with associated weights.
A document is classified as the type whose patterns accumulate the highest score,
provided that score meets _CLASSIFICATION_THRESHOLD.

This module is intentionally standalone -- it only depends on ``re`` and
``logging`` so it can be used inside an AWS Lambda without extra dependencies.
"""

import logging
import re

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weighted regex patterns per document type (pre-compiled for performance)
# ---------------------------------------------------------------------------

_RAW_PATTERNS: dict[str, list[tuple[str, int]]] = {
    "aadhaar": [
        (r"(?i)unique\s*identification\s*authority", 10),
        (r"(?i)\buidai\b", 10),
        (r"(?i)aadhaar", 8),
        (r"आधार", 8),
        (r"\b\d{4}\s?\d{4}\s?\d{4}\b", 6),
        (r"(?i)vid\s*:?\s*\d{16}", 5),
        (r"(?i)(download\s*date|issue\s*date|issued)", 4),
        (r"(?i)go[uv]e[rn]?n?ment\s*of\s*india", 2),
        (r"(?i)proof\s*of\s*identity", 6),
        (r"(?i)\bDOB\b", 3),
    ],
    "pan": [
        (r"(?i)income\s*tax", 10),
        (r"(?i)permanent\s*account\s*number", 10),
        (r"\b[A-Z]{5}\d{4}[A-Z]\b", 8),
        (r"(?i)govt\.?\s*of\s*india", 2),
    ],
    "passport": [
        (r"(?i)republic\s*of\s*india", 8),
        (r"P<IND", 10),
        (r"(?i)\bpassport\b", 8),
        (r"भारत\s*गणराज्य", 8),
        (r"(?i)पासपोर्ट", 6),
        (r"\b[A-Z]{1,2}\d{6,7}\b", 5),
    ],
    "voter": [
        (r"(?i)election\s*commission", 8),
        (r"(?i)\bepic\b", 8),
        (r"(?i)elector", 6),
        (r"\b[A-Z]{3}\d{7}\b", 6),
    ],
    "e_voter": [
        (r"(?i)election\s*commission", 8),
        (r"(?i)\bepic\b", 8),
        (r"\b[A-Z]{3}\d{7}\b", 6),
        (r"(?i)e[\-\s]*epic", 10),
        (r"(?i)digital\s*voter", 10),
    ],
}

# Pre-compile all patterns at module load time
_CLASSIFICATION_PATTERNS: dict[str, list[tuple[re.Pattern, int]]] = {
    doc_type: [(re.compile(pattern), weight) for pattern, weight in patterns]
    for doc_type, patterns in _RAW_PATTERNS.items()
}

_CLASSIFICATION_THRESHOLD = 8


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_document(ocr_text: str) -> str:
    """Classify document type from OCR text using weighted pattern matching.

    Returns one of: 'aadhaar', 'pan', 'passport', 'voter', 'e_voter', 'generic'.
    """
    scores: dict[str, int] = {}
    for doc_type, patterns in _CLASSIFICATION_PATTERNS.items():
        score = 0
        for compiled_re, weight in patterns:
            if compiled_re.search(ocr_text):
                score += weight
        scores[doc_type] = score

    # E_VOTER vs VOTER disambiguation: both share base patterns.
    # Only pick e_voter if e_voter-specific patterns (weight 10) contributed.
    if scores.get("e_voter", 0) > scores.get("voter", 0):
        best_type = "e_voter"
        best_score = scores["e_voter"]
    elif scores.get("voter", 0) >= _CLASSIFICATION_THRESHOLD:
        best_type = "voter"
        best_score = scores["voter"]
    else:
        # Pick the highest-scoring type (excluding voter/e_voter already handled)
        best_type = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best_type]

    if best_score < _CLASSIFICATION_THRESHOLD:
        log.info(
            "  Classification: generic (no type scored >= %d, scores: %s)",
            _CLASSIFICATION_THRESHOLD,
            scores,
        )
        return "generic"

    log.info(
        "  Classification: %s (score: %d, all: %s)",
        best_type,
        best_score,
        scores,
    )
    return best_type


def is_aadhaar(ocr_text: str) -> bool:
    """Return True if *ocr_text* is classified as an Aadhaar document.

    For speed, this evaluates only the aadhaar patterns rather than scoring
    every document type.  If the aadhaar score alone meets the threshold the
    function returns ``True`` immediately.
    """
    score = 0
    for compiled_re, weight in _CLASSIFICATION_PATTERNS["aadhaar"]:
        if compiled_re.search(ocr_text):
            score += weight
            # Short-circuit once the threshold is reached.
            if score >= _CLASSIFICATION_THRESHOLD:
                return True
    return False
