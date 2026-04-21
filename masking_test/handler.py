"""
AWS Lambda handler: Aadhaar ONNX inference + validation.

Supports three event sources:
  1. S3 trigger: event["Records"] from S3 (bucket + key taken from first record).
  2. Direct invoke with "s3_bucket" + "s3_key": image read from S3.
  3. Direct invoke with "body_base64": base64-encoded image bytes.

Optional: "model_s3_uri" in event to load model from a specific S3 URI.
When image is from S3 (trigger or s3_bucket+s3_key), the model is read from the
same bucket at key model/best_new.onnx unless model_s3_uri or MODEL_PATH is set.

Environment:
  Layer adds /opt/python automatically; do not set PYTHONPATH=/opt.
  MODEL_PATH: local path to .onnx if bundled in zip (default /var/task/best_new.onnx)
  MODEL_S3_KEY: S3 key when using same-bucket model (default model/best_new.onnx)
  CONF: confidence threshold (default 0.55, match prediction.py).
  MODEL_CONF: detection filter threshold (default 0.55, match prediction.py).
  PDF_DPI: DPI for PDF page rendering (default 150, match prediction.py).
  IOU_THRESH, MAX_DET: NMS params (defaults 0.7, 300).
  For best accuracy parity with prediction.py, use same CONF/MODEL_CONF (0.55) and PDF_DPI (150).
  VALIDATE_DEBUG=1 : log validation points, detected/missing classes, scores (CloudWatch).
  AADHAAR_METADATA_TABLE: if set, write processing metadata to this DynamoDB table for dashboard.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any
import re
import uuid
from urllib.parse import unquote, unquote_plus

import numpy as np

logger = logging.getLogger(__name__)

# Class names in model order (same as data.yaml)
CLASS_NAMES = [
    "aadhaar_address",
    "aadhaar_dob",
    "aadhaar_gender",
    "aadhaar_holder_name",
    "aadhaar_logo",
    "aadhaar_no",
    "aadhaar_photo",
    "aadhaar_qr",
    "aadhar_no_mask",
    "emblem",
    "gov_logo",
]

# Validation rules (must match validate_aadhaar.py)
FRONT_REQUIRED = {"aadhaar_no"}
FRONT_OPTIONAL = {"aadhaar_dob", "emblem", "aadhaar_logo", "aadhaar_qr"}
BACK_REQUIRED = {"aadhaar_address", "aadhaar_logo"}
BACK_BONUS = {"aadhaar_qr", "aadhar_no_mask", "emblem", "gov_logo", "aadhaar_logo"}
BACK_OPTIONAL = {"emblem", "gov_logo"}
FULL_REQUIRED = FRONT_REQUIRED | BACK_REQUIRED
ALL_AADHAAR = set(CLASS_NAMES)
VALID_THRESHOLD = 70.0
UNCERTAIN_THRESHOLD = 40.0

IMGSZ = 640
# Match prediction.py: CONF default 0.55 (validation + masking threshold)
CONF_THRESHOLD = float(os.environ.get("CONF", "0.40"))
# Model detection threshold (match prediction.py model.predict(conf=0.55))
# If Lambda classifies front pages as back, set MODEL_CONF=0.45 in env to allow more detections.
MODEL_CONF = float(os.environ.get("MODEL_CONF", "0.40"))
# NMS: match Ultralytics defaults (iou=0.7, max_det=300)
IOU_THRESH = float(os.environ.get("IOU_THRESH", "0.7"))
MAX_DET = int(os.environ.get("MAX_DET", "300"))
# PDF rendering DPI (match prediction.py PDF_DPI=150)
PDF_DPI = int(os.environ.get("PDF_DPI", "150"))
# Per-class NMS by default (agnostic_nms=False in Ultralytics default.yaml)
AGNOSTIC_NMS = os.environ.get("AGNOSTIC_NMS", "false").lower() in ("1", "true", "yes")
# Set VALIDATE_DEBUG=1 to log validation points, detected/missing, scores (CloudWatch)
DEBUG = os.environ.get("VALIDATE_DEBUG", "").strip().lower() in ("1", "true", "yes")

# Classes to blacken in the masked image (same as mask_aadhaar.py)
MASK_CLASSES = {"aadhaar_qr", "aadhar_no_mask"}

# How many of the 12 Aadhaar digits to mask (8 = last 4 visible, 12 = fully masked)
MASK_DIGITS = int(os.environ.get("MASK_DIGITS", "8"))

# OCR-based fallback: when YOLO misses an Aadhaar number, OCR can catch it
OCR_FALLBACK = os.environ.get("OCR_FALLBACK", "true").lower() in ("1", "true", "yes")

# Global session (reused across invocations)
_session = None
_model_path_loaded: str | None = None
_ocr_engine = None


def _score(detected: set[str], required: set[str], bonus: set[str] | None = None) -> float:
    if not required:
        return 0.0
    base = len(detected & required) / len(required) * 100.0
    if bonus:
        base = min(100.0, base + len(detected & bonus) * 10.0)
    return base


def _verdict(score: float) -> str:
    if score >= VALID_THRESHOLD:
        return "VALID"
    if score >= UNCERTAIN_THRESHOLD:
        return "UNCERTAIN"
    return "INVALID"


def _reason(
    verdict: str,
    card_type: str,
    missing: list[str],
    missing_bonus: list[str] | None = None,
) -> str:
    """Human-readable reason (match validate_aadhaar._reason)."""
    if verdict == "VALID":
        extra = ""
        if missing_bonus:
            extra = f" (optional not detected: {', '.join(missing_bonus)})"
        return f"Valid {card_type}-side Aadhaar detected.{extra}"
    if verdict == "UNCERTAIN":
        return (
            f"Partial {card_type}-side match. "
            f"Missing required: {', '.join(missing) if missing else 'none'}. "
            "Card may be partially visible, blurry, or cropped."
        )
    return (
        f"Too few {card_type}-side classes detected. "
        f"Missing required: {', '.join(missing) if missing else 'none'}."
    )


def validate_from_detections(
    detected_classes: set[str],
    conf_threshold: float = 0.5,
    source_label: str | None = None,
) -> dict:
    """Validation result from set of detected class names (no Ultralytics). Matches validate_aadhaar.validate() return shape."""
    aadhaar_hits = detected_classes & ALL_AADHAAR
    base_invalid = {
        "is_aadhaar": False,
        "card_type": "unknown",
        "validation": "INVALID",
        "score": 0.0,
        "detected_classes": sorted(detected_classes),
        "missing_required": [],
        "missing_bonus": [],
        "masked_classes": [],
        "reason": "No Aadhaar-specific classes detected.",
        "confidence_threshold_used": conf_threshold,
    }
    if source_label is not None:
        base_invalid["file"] = source_label
    if not aadhaar_hits:
        return base_invalid
    front_score = _score(detected_classes, FRONT_REQUIRED)
    back_score = _score(detected_classes, BACK_REQUIRED, bonus=BACK_BONUS)
    full_score = _score(detected_classes, FULL_REQUIRED)
    if full_score >= VALID_THRESHOLD:
        card_type, required, score = "full", FULL_REQUIRED, full_score
    elif front_score >= back_score:
        card_type, required, score = "front", FRONT_REQUIRED, front_score
    else:
        card_type, required, score = "back", BACK_REQUIRED, back_score
    missing = sorted(required - detected_classes)
    missing_bonus = sorted(BACK_BONUS - detected_classes) if card_type == "back" else []
    verdict = _verdict(score)
    reason = _reason(verdict, card_type, missing, missing_bonus)
    logger.info("[validate] %s card_type=%s verdict=%s score=%.1f",
                source_label or "page", card_type, verdict, score)
    out = {
        "is_aadhaar": True,
        "card_type": card_type,
        "validation": verdict,
        "score": round(score, 1),
        "detected_classes": sorted(detected_classes),
        "missing_required": missing,
        "missing_bonus": missing_bonus,
        "confidence_threshold_used": conf_threshold,
        "reason": reason,
    }
    if source_label is not None:
        out["file"] = source_label
    return out


def get_session(model_path: str):
    global _session, _model_path_loaded
    import onnxruntime as ort
    if _session is None or _model_path_loaded != model_path:
        _session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        _model_path_loaded = model_path
    return _session


def _is_pdf(raw: bytes, s3_key: str | None) -> bool:
    """True if content looks like PDF (magic or filename)."""
    if raw[:4] == b"%PDF":
        return True
    if s3_key and s3_key.lower().endswith(".pdf"):
        return True
    return False


def _load_raw_from_event(event: dict) -> tuple[bytes, str | None]:
    """Get raw bytes and optional s3_key from event."""
    if "body_base64" in event:
        raw = base64.b64decode(event["body_base64"])
        return raw, None
    if "s3_bucket" in event and "s3_key" in event:
        import boto3
        from botocore.exceptions import ClientError
        s3 = boto3.client("s3")
        bucket = event["s3_bucket"]
        key = event["s3_key"]
        try:
            body = s3.get_object(Bucket=bucket, Key=key)["Body"]
            raw = body.read()
            return raw, key
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "NoSuchKey" and "+" in key:
                # Retry with '+' as space (object may have been stored with spaces in the key)
                alt_key = key.replace("+", " ")
                logger.info("NoSuchKey for key with '+', retrying with spaces: %s", alt_key)
                body = s3.get_object(Bucket=bucket, Key=alt_key)["Body"]
                raw = body.read()
                return raw, alt_key
            raise
    raise ValueError("Provide body_base64 or s3_bucket+s3_key")


def load_images_from_event(event: dict) -> tuple[bool, list[np.ndarray], str | None]:
    """
    Load one or more images from event (S3 or base64).
    Returns (is_pdf, list of RGB numpy images, source_key or None).
    For PDF: all pages as numpy arrays. For image: single-element list.
    """
    from PIL import Image
    raw, s3_key = _load_raw_from_event(event)
    if _is_pdf(raw, s3_key):
        import fitz
        doc = fitz.open(stream=raw, filetype="pdf")
        if len(doc) == 0:
            doc.close()
            raise ValueError("PDF has no pages")
        images = []
        for i in range(len(doc)):
            page = doc[i]
            pix = page.get_pixmap(dpi=PDF_DPI)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            # Keep as RGB (preprocess will do RGB->BGR). Match pdf_pages.py channel handling.
            if pix.n == 4:
                img = img[:, :, :3].copy()  # RGBA -> RGB (drop alpha)
            elif pix.n == 1:
                gray = np.squeeze(img)
                img = np.stack([gray, gray, gray], axis=-1)  # grayscale -> 3ch RGB
            # else: already RGB (pix.n == 3)
            images.append(img)
        doc.close()
        return True, images, s3_key
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    return False, [np.array(pil)], s3_key


def preprocess(img: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int, int, int]]:
    """
    Letterbox resize to IMGSZ, normalize [0,1], NCHW. Matches Ultralytics LetterBox:
    - Input: RGB numpy HWC (from PyMuPDF or Pillow).
    - Pad value 114, center padding, HWC->NCHW, /255.
    - Ultralytics ONNX expects RGB input (BGR->RGB is done by Ultralytics before ONNX export).
    - scaleup: env SCALEUP=true (default) allows scaling up small images; false = only scale down (like val).
    """
    from PIL import Image
    h, w = img.shape[:2]
    r = min(IMGSZ / h, IMGSZ / w)
    scaleup = os.environ.get("SCALEUP", "true").lower() in ("1", "true", "yes")
    if not scaleup:
        r = min(r, 1.0)
    new_h, new_w = int(round(h * r)), int(round(w * r))
    pil = Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR)
    resized = np.array(pil)
    # Same padding split as Ultralytics LetterBox: round(dw/2 - 0.1), round(dw/2 + 0.1)
    dw = IMGSZ - new_w
    dh = IMGSZ - new_h
    pad_left = int(round(dw / 2.0 - 0.1))
    pad_right = int(round(dw / 2.0 + 0.1))
    pad_top = int(round(dh / 2.0 - 0.1))
    pad_bottom = int(round(dh / 2.0 + 0.1))
    padded = np.full((IMGSZ, IMGSZ, 3), 114, dtype=np.uint8)
    padded[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    # RGB input -> NCHW float32 [0,1]. No BGR flip: Ultralytics ONNX expects RGB.
    x = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    return x, r, (pad_left, pad_top, w, h)


def _nms_boxes(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, max_det: int
) -> np.ndarray:
    """Non-maximum suppression (numpy). Returns indices of kept boxes (sorted by score desc)."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores)
    keep: list[int] = []
    while len(order) > 0 and len(keep) < max_det:
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_threshold]
    return np.array(keep, dtype=np.int64)


def _nms_per_class(
    boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray,
    iou_threshold: float, max_det: int,
) -> np.ndarray:
    """
    Per-class NMS (agnostic_nms=False). Run NMS separately per class, then merge and
    take top max_det by score. Matches Ultralytics default behavior.
    Returns indices into (boxes, scores, class_ids).
    """
    out: list[int] = []
    for cid in np.unique(class_ids):
        mask = class_ids == cid
        inds = np.where(mask)[0]
        if len(inds) == 0:
            continue
        keep_c = _nms_boxes(
            boxes[inds], scores[inds], iou_threshold, max_det=max_det
        )
        out.extend(inds[keep_c].tolist())
    # Sort by score descending and cap at max_det (match Ultralytics i[:max_det])
    out = sorted(out, key=lambda i: -scores[i])[:max_det]
    return np.array(out, dtype=np.int64)


def postprocess(
    output: np.ndarray, model_conf: float, iou_thresh: float, max_det: int,
    agnostic_nms: bool = False,
) -> tuple[set[str], list[tuple[float, float, float, float, str, float]], list[dict]]:
    """
    YOLO output (1, 15, 8400) -> detected class names, mask boxes with scores,
    and full detection details list.
    """
    pred = output[0]  # (15, 8400)
    cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
    x1_lb = cx - w / 2.0
    y1_lb = cy - h / 2.0
    x2_lb = cx + w / 2.0
    y2_lb = cy + h / 2.0
    scores = pred[4:].max(axis=0)
    class_ids = pred[4:].argmax(axis=0)
    cand = scores >= model_conf
    if not np.any(cand):
        return set(), [], []
    idx = np.where(cand)[0]
    boxes_np = np.stack([x1_lb[idx], y1_lb[idx], x2_lb[idx], y2_lb[idx]], axis=1)
    scores_np = scores[idx]
    class_ids_np = class_ids[idx]
    if agnostic_nms:
        keep = _nms_boxes(boxes_np, scores_np, iou_thresh, max_det)
    else:
        keep = _nms_per_class(boxes_np, scores_np, class_ids_np, iou_thresh, max_det)
    detected = set()
    mask_boxes: list[tuple[float, float, float, float, str, float]] = []
    all_detections: list[dict] = []
    for i in keep:
        cid = int(class_ids_np[i])
        if 0 <= cid < len(CLASS_NAMES):
            name = CLASS_NAMES[cid]
            conf = float(scores_np[i])
            box = (float(boxes_np[i, 0]), float(boxes_np[i, 1]),
                   float(boxes_np[i, 2]), float(boxes_np[i, 3]))
            all_detections.append({
                "class": name,
                "confidence": round(conf, 4),
                "box": box,
                "kept": conf >= model_conf,
                "masked": name in MASK_CLASSES and conf >= CONF_THRESHOLD,
            })
            if conf >= model_conf:
                detected.add(name)
            if name in MASK_CLASSES and conf >= CONF_THRESHOLD:
                mask_boxes.append((*box, name, conf))
    return detected, mask_boxes, all_detections


def expand_mask_boxes_for_digits(
    mask_boxes: list[tuple[float, float, float, float, str, float]],
    mask_digits: int,
    img_width: float = 0,
) -> list[tuple[float, float, float, float, str, float]]:
    """
    When mask_digits >= 12, expand `aadhar_no_mask` boxes rightward so the
    black rectangle covers the full 12-digit number instead of only ~8 digits.
    The YOLO model's `aadhar_no_mask` bounding box typically covers only the
    first 8 digits; we extend x2 by ~50 % of the box width (8→12 = factor 1.5).
    """
    if mask_digits < 12:
        return mask_boxes
    expanded: list[tuple[float, float, float, float, str, float]] = []
    for (x1, y1, x2, y2, name, conf) in mask_boxes:
        if name == "aadhar_no_mask":
            box_w = x2 - x1
            # Aadhaar format "XXXX XXXX XXXX": YOLO covers ~8 digits + 1 space,
            # extend by 60% to cover remaining 4 digits + 1 space with margin
            new_x2 = x2 + box_w * 0.6
            if img_width > 0:
                new_x2 = min(new_x2, img_width)
            expanded.append((x1, y1, new_x2, y2, name, conf))
        else:
            expanded.append((x1, y1, x2, y2, name, conf))
    return expanded


def postprocess_low_conf(
    output: np.ndarray, low_thresh: float = 0.15,
) -> list[dict]:
    """Return ALL detections above a very low threshold — for debugging missed detections."""
    pred = output[0]
    cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
    x1_lb = cx - w / 2.0
    y1_lb = cy - h / 2.0
    x2_lb = cx + w / 2.0
    y2_lb = cy + h / 2.0
    scores = pred[4:].max(axis=0)
    class_ids = pred[4:].argmax(axis=0)
    cand = scores >= low_thresh
    if not np.any(cand):
        return []
    idx = np.where(cand)[0]
    results = []
    for i in idx:
        cid = int(class_ids[i])
        if 0 <= cid < len(CLASS_NAMES):
            results.append({
                "class": CLASS_NAMES[cid],
                "confidence": round(float(scores[i]), 4),
                "box": (float(x1_lb[i]), float(y1_lb[i]), float(x2_lb[i]), float(y2_lb[i])),
            })
    results.sort(key=lambda d: -d["confidence"])
    return results


def _letterbox_to_orig(
    x1: float, y1: float, x2: float, y2: float,
    scale: float, pad_x: int, pad_y: int,
) -> tuple[int, int, int, int]:
    """Convert letterbox coords to original image coords (clamped)."""
    x1_o = (x1 - pad_x) / scale
    y1_o = (y1 - pad_y) / scale
    x2_o = (x2 - pad_x) / scale
    y2_o = (y2 - pad_y) / scale
    return int(max(0, x1_o)), int(max(0, y1_o)), int(max(0, x2_o)), int(max(0, y2_o))


def _orig_to_letterbox(
    x1: int, y1: int, x2: int, y2: int,
    scale: float, pad_x: int, pad_y: int,
) -> tuple[float, float, float, float]:
    """Convert original image coords back to letterbox coords (for appending OCR boxes to YOLO list)."""
    return (x1 * scale + pad_x, y1 * scale + pad_y,
            x2 * scale + pad_x, y2 * scale + pad_y)


# ── Verhoeff checksum (validates 12-digit Aadhaar numbers) ───────────────
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


def _verhoeff_check(number_str: str) -> bool:
    """Return True if the digit string passes the Verhoeff checksum."""
    try:
        c = 0
        for i, ch in enumerate(reversed(number_str)):
            c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
        return c == 0
    except (ValueError, IndexError):
        return False


def _iou(box1: tuple, box2: tuple) -> float:
    """Intersection over Union of two (x1,y1,x2,y2) boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def ocr_fallback_mask(
    img: np.ndarray,
    existing_mask_boxes_orig: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """
    OCR-based fallback: find Aadhaar numbers in the image that YOLO missed.

    Runs lightweight OCR, finds 12-digit numbers passing Verhoeff checksum,
    returns bounding boxes for any not already covered by YOLO masks.
    """
    global _ocr_engine

    if not OCR_FALLBACK:
        return []

    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        logger.info("[OCR fallback] rapidocr not available, skipping")
        return []

    if _ocr_engine is None:
        _ocr_engine = RapidOCR()

    result, _ = _ocr_engine(img)
    if not result:
        logger.info("[OCR fallback] no text detected")
        return []

    aadhaar_re = re.compile(r"\d{4}\s?\d{4}\s?\d{4}")
    fallback_boxes: list[tuple[int, int, int, int]] = []

    for bbox, text, conf in result:
        if not aadhaar_re.search(text):
            continue

        digits = re.sub(r"\D", "", text)
        if len(digits) != 12:
            continue

        if not _verhoeff_check(digits):
            logger.debug("[OCR fallback] skipping '%s' — failed Verhoeff", text)
            continue

        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        full_box = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

        already_covered = any(
            _iou(full_box, existing[:4]) > 0.3
            for existing in existing_mask_boxes_orig
        )

        if already_covered:
            logger.info("[OCR fallback] Aadhaar '%s' already masked by YOLO", digits[-4:])
        else:
            x1, y1, x2, y2 = full_box
            width = x2 - x1
            if MASK_DIGITS >= 12:
                partial_box = full_box
            else:
                mask_x2 = int(x1 + width * MASK_DIGITS / 12)
                partial_box = (x1, y1, mask_x2, y2)
            logger.info(
                "[OCR fallback] FOUND unmasked Aadhaar 'XXXX XXXX %s' at %s → masking %d/12 digits: %s (ocr_conf=%.2f)",
                digits[-4:], full_box, MASK_DIGITS, partial_box, float(conf),
            )
            fallback_boxes.append(partial_box)

    logger.info("[OCR fallback] %d additional region(s) to mask", len(fallback_boxes))
    return fallback_boxes


def apply_mask_pillow(
    img: np.ndarray, boxes_orig: list[tuple[int, int, int, int]]
) -> bytes:
    """Draw black rectangles on image (RGB numpy HWC), return JPEG bytes."""
    from PIL import Image, ImageDraw
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    for (x1, y1, x2, y2) in boxes_orig:
        draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _build_masked_pdf_bytes(
    original_pdf_path: str, page_to_masked_image_path: dict[int, str]
) -> bytes:
    """
    Build a PDF from the original, replacing given pages with masked images.
    page_to_masked_image_path: 1-based page number -> path to masked image (JPG/PNG).
    Returns PDF bytes.
    """
    import fitz
    src = fitz.open(original_pdf_path)
    out_doc = fitz.open()
    try:
        for i in range(len(src)):
            page_num = i + 1
            if page_num in page_to_masked_image_path:
                mask_path = page_to_masked_image_path[page_num]
                img_doc = fitz.open(mask_path)
                img_page = img_doc[0]
                w, h = img_page.rect.width, img_page.rect.height
                img_doc.close()
                out_page = out_doc.new_page(width=w, height=h)
                out_page.insert_image(out_page.rect, filename=mask_path)
            else:
                out_doc.insert_pdf(src, from_page=i, to_page=i)
        buf = io.BytesIO()
        out_doc.save(buf)
        return buf.getvalue()
    finally:
        out_doc.close()
        src.close()


def save_masked_to_s3(
    bucket: str, source_key: str, img: np.ndarray,
    mask_boxes_letterbox: list[tuple[float, float, float, float, str]],
    scale: float, pad_x: int, pad_y: int,
) -> str | None:
    """
    Single image: draw mask and upload mask/<basename>_masked.jpg.
    Returns the S3 key written, or None if nothing uploaded.
    """
    if not mask_boxes_letterbox:
        return None
    boxes_orig = [
        _letterbox_to_orig(x1, y1, x2, y2, scale, pad_x, pad_y)
        for (x1, y1, x2, y2, *_rest) in mask_boxes_letterbox
    ]
    jpeg_bytes = apply_mask_pillow(img, boxes_orig)
    base = _safe_mask_basename(source_key)
    prefix = _get_output_prefix(source_key)
    import boto3
    s3 = boto3.client("s3")
    mask_key = f"{prefix}/{base}_masked.jpg"
    s3.put_object(
        Bucket=bucket,
        Key=mask_key,
        Body=jpeg_bytes,
        ContentType="image/jpeg",
    )
    return mask_key


def save_masked_pdf_to_s3(
    bucket: str, source_key: str,
    page_data: list[tuple[int, np.ndarray, list[tuple[float, float, float, float, str]], float, int, int]],
) -> str | None:
    """
    Build masked PDF from original: replace each given page with its masked image, upload to mask/<basename>_masked.pdf.
    page_data: list of (page_num_1based, img, mask_boxes_letterbox, scale, pad_x, pad_y) for VALID pages only.
    Returns the S3 key written, or None if page_data is empty.
    """
    if not page_data:
        return None
    import boto3
    s3 = boto3.client("s3")
    base = _safe_mask_basename(source_key)
    prefix = _get_output_prefix(source_key)
    orig_pdf = "/tmp/lambda_orig.pdf"
    s3.download_file(bucket, source_key, orig_pdf)
    page_to_masked_path: dict[int, str] = {}
    for page_num, img, mask_boxes_letterbox, scale, pad_x, pad_y in page_data:
        if not mask_boxes_letterbox:
            continue
        boxes_orig = [
            _letterbox_to_orig(x1, y1, x2, y2, scale, pad_x, pad_y)
            for (x1, y1, x2, y2, *_rest) in mask_boxes_letterbox
        ]
        jpeg_bytes = apply_mask_pillow(img, boxes_orig)
        path = f"/tmp/lambda_masked_p{page_num}.jpg"
        with open(path, "wb") as f:
            f.write(jpeg_bytes)
        page_to_masked_path[page_num] = path
    if not page_to_masked_path:
        return None
    pdf_bytes = _build_masked_pdf_bytes(orig_pdf, page_to_masked_path)
    mask_key = f"{prefix}/{base}_masked.pdf"
    s3.put_object(
        Bucket=bucket,
        Key=mask_key,
        Body=pdf_bytes,
        ContentType="application/pdf",
    )
    return mask_key


def _safe_mask_basename(source_key: str) -> str:
    """Basename without extension, sanitized for S3 key (no spaces or special chars)."""
    base = os.path.splitext(os.path.basename(source_key))[0]
    # Replace any character that isn't alphanumeric, underscore, or hyphen with underscore
    return re.sub(r"[^\w\-]", "_", base or "masked")


def _get_output_prefix(source_key: str) -> str:
    """Route output based on input prefix: chola_input/ → chola_result/, else mask/."""
    if source_key and source_key.startswith("chola_input/"):
        return "chola_result"
    return "mask"


def _write_metadata(
    source_key: str | None,
    s3_bucket: str | None,
    out: dict,
) -> None:
    """Write processing metadata to DynamoDB if AADHAAR_METADATA_TABLE is set."""
    table_name = os.environ.get("AADHAAR_METADATA_TABLE", "").strip()
    if not table_name or not source_key or not s3_bucket:
        return
    try:
        import boto3
        validation = out.get("validation") or {}
        sk = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        # Your table uses id (Number) as key; DynamoDB Number must be string in API (max 38 digits)
        numeric_id = uuid.uuid4().int % (10**38)
        item = {
            "id": {"N": str(numeric_id)},
            "pk": {"S": "RECORD"},
            "sk": {"S": sk},
            "s3_bucket": {"S": s3_bucket},
            "s3_key": {"S": source_key},
            "card_type": {"S": str(validation.get("card_type", ""))},
            "validation": {"S": str(validation.get("validation", ""))},
            "score": {"N": str(validation.get("score", 0))},
            "page_count": {"N": str(len(out.get("pages") or []))},
            "valid_pages": {"N": str(len(out.get("valid_pages") or []))},
            "latency_ms": {"M": {k: {"N": str(v)} for k, v in (out.get("latency_ms") or {}).items()}},
        }
        if out.get("mask_output_key"):
            item["mask_output_key"] = {"S": out["mask_output_key"]}
        detected = validation.get("detected_classes")
        if isinstance(detected, list):
            item["detected_classes"] = {"L": [{"S": str(x)} for x in detected]}
        boto3.client("dynamodb").put_item(TableName=table_name, Item=item)
    except Exception as e:
        logger.warning("Failed to write metadata to DynamoDB: %s", e)


def _normalize_event(event: dict) -> tuple[dict, str | None]:
    """
    Normalize event so it always has s3_bucket + s3_key or body_base64.
    If event is from S3 trigger (Records), extract bucket and key from first record.
    Also reads mask_digits from S3 object user-metadata and updates the global MASK_DIGITS.
    Returns (event_for_load_image, source_key_or_none).
    """
    global MASK_DIGITS
    if event.get("Records") and len(event["Records"]) > 0:
        record = event["Records"][0]
        if "s3" in record:
            bucket = record["s3"]["bucket"]["name"]
            raw_key = record["s3"]["object"]["key"]
            key = unquote_plus(raw_key) if "+" in raw_key else unquote(raw_key)

            try:
                import boto3
                head = boto3.client("s3").head_object(Bucket=bucket, Key=key)
                user_meta = head.get("Metadata", {})
                if "mask_digits" in user_meta:
                    MASK_DIGITS = int(user_meta["mask_digits"])
                    logger.info("[normalize] mask_digits from S3 metadata: %d", MASK_DIGITS)
            except Exception as exc:
                logger.warning("[normalize] Could not read S3 metadata: %s", exc)

            return {"s3_bucket": bucket, "s3_key": key, **event}, key
    return event, None


def load_model_from_s3(uri: str) -> str:
    """Download model from S3 to /tmp and return local path."""
    import boto3
    from urllib.parse import urlparse
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    bucket, key = parsed.netloc, parsed.path.lstrip("/")
    local = f"/tmp/{os.path.basename(key)}"
    boto3.client("s3").download_file(bucket, key, local)
    return local


def _resolve_model_path(event: dict) -> str:
    """Resolve model to a local path: from event URI, same-bucket model/ folder, or env MODEL_PATH."""
    if event.get("model_s3_uri"):
        return load_model_from_s3(event["model_s3_uri"])
    bucket = event.get("s3_bucket")
    if bucket:
        # Same bucket, model folder: s3://bucket/model/best_new.onnx
        model_key = os.environ.get("MODEL_S3_KEY", "model/best_new.onnx")
        return load_model_from_s3(f"s3://{bucket}/{model_key}")
    return os.environ.get("MODEL_PATH", "/var/task/best_new.onnx")


def lambda_handler(event: dict, context: Any) -> dict:
    # Ensure INFO logs appear in CloudWatch (Lambda default is often WARNING)
    logging.getLogger().setLevel(logging.INFO)
    logger.setLevel(logging.INFO)

    t0 = time.perf_counter()
    event, source_key = _normalize_event(event)
    logger.info("handler invoked source_key=%s MASK_DIGITS=%d CONF=%.2f",
                source_key or "(body_base64)", MASK_DIGITS, CONF_THRESHOLD)
    model_path = _resolve_model_path(event)
    session = get_session(model_path)
    input_name = session.get_inputs()[0].name

    is_pdf, images, _ = load_images_from_event(event)
    logger.info("loaded %d page(s)", len(images))

    page_results: list[tuple[int, dict, list, np.ndarray, float, int, int]] = []
    t_preprocess, t_inference, t_postprocess, t_ocr_fallback = 0.0, 0.0, 0.0, 0.0
    total_pages = len(images)

    for page_idx, img in enumerate(images):
        page_num = page_idx + 1
        logger.info("page %s/%s", page_num, total_pages)
        t_a = time.perf_counter()
        x, scale, (pad_x, pad_y, _orig_w, _orig_h) = preprocess(img)
        page_preprocess_ms = (time.perf_counter() - t_a) * 1000
        t_preprocess += page_preprocess_ms / 1000

        t_b = time.perf_counter()
        outputs = session.run(None, {input_name: x})
        page_inference_ms = (time.perf_counter() - t_b) * 1000
        t_inference += page_inference_ms / 1000

        t_c = time.perf_counter()
        detected, mask_boxes_letterbox, _ = postprocess(
            outputs[0], MODEL_CONF, IOU_THRESH, MAX_DET,
            agnostic_nms=AGNOSTIC_NMS,
        )
        mask_boxes_letterbox = expand_mask_boxes_for_digits(
            mask_boxes_letterbox, MASK_DIGITS, img_width=float(IMGSZ),
        )
        page_postprocess_ms = (time.perf_counter() - t_c) * 1000

        source_label = (f"{source_key} p.{page_num}" if source_key else f"page_{page_num}")
        validation = validate_from_detections(detected, CONF_THRESHOLD, source_label=source_label)

        masked_cls = sorted({name for *_, name in mask_boxes_letterbox})
        validation["masked_classes"] = masked_cls

        # OCR fallback: catch Aadhaar numbers YOLO missed
        page_ocr_fb_ms = 0.0
        if OCR_FALLBACK:
            t_fb = time.perf_counter()
            yolo_boxes_orig = [
                _letterbox_to_orig(x1, y1, x2, y2, scale, pad_x, pad_y)
                for (x1, y1, x2, y2, *_rest) in mask_boxes_letterbox
            ]
            ocr_extra = ocr_fallback_mask(img, yolo_boxes_orig)
            if ocr_extra:
                for box_orig in ocr_extra:
                    lb = _orig_to_letterbox(*box_orig, scale, pad_x, pad_y)
                    mask_boxes_letterbox.append((*lb, "ocr_fallback_aadhaar_no", 1.0))
                validation["ocr_fallback_count"] = len(ocr_extra)
                masked_cls = sorted({name for *_, name in mask_boxes_letterbox})
                validation["masked_classes"] = masked_cls
            page_ocr_fb_ms = (time.perf_counter() - t_fb) * 1000
            t_ocr_fallback += page_ocr_fb_ms / 1000

        t_postprocess += page_postprocess_ms / 1000

        logger.info(
            "[handler] page %s latency: preprocess=%.1fms, inference=%.1fms, postprocess=%.1fms, ocr_fallback=%.1fms",
            page_num, page_preprocess_ms, page_inference_ms, page_postprocess_ms, page_ocr_fb_ms,
        )

        page_results.append((page_num, validation, mask_boxes_letterbox, img, scale, pad_x, pad_y))

    t3 = time.perf_counter()

    # Masked output: PDF (all valid pages) or single image
    t_save_start = time.perf_counter()
    mask_output_key = None
    if event.get("s3_bucket") and source_key is not None:
        if is_pdf and source_key.lower().endswith(".pdf"):
            valid_pages_data = [
                (pnum, img, boxes, scale, px, py)
                for pnum, val, boxes, img, scale, px, py in page_results
                if val.get("validation") == "VALID" and boxes
            ]
            mask_output_key = save_masked_pdf_to_s3(event["s3_bucket"], source_key, valid_pages_data)
        elif not is_pdf and len(page_results) == 1:
            pnum, val, boxes, img, scale, px, py = page_results[0]
            if val.get("validation") == "VALID" and boxes:
                mask_output_key = save_masked_to_s3(
                    event["s3_bucket"], source_key, img, boxes, scale, px, py
                )
    save_ms = round((time.perf_counter() - t_save_start) * 1000, 1)
    total_ms = round((time.perf_counter() - t0) * 1000, 1)

    logger.info(
        "[handler] TOTAL latency: preprocess=%.1fms, inference=%.1fms, postprocess=%.1fms, "
        "ocr_fallback=%.1fms, s3_save=%.1fms, total=%.1fms (%d pages)",
        round(t_preprocess * 1000, 1), round(t_inference * 1000, 1),
        round(t_postprocess * 1000, 1), round(t_ocr_fallback * 1000, 1),
        save_ms, total_ms, total_pages,
    )

    # Backward compat: first page validation; add per-page and summary
    validation = page_results[0][1] if page_results else {}
    pages = [{"page": pnum, "validation": val} for pnum, val, *_ in page_results]
    valid_pages = [pnum for pnum, val, *_ in page_results if val.get("validation") == "VALID"]

    out = {
        "statusCode": 200,
        "validation": validation,
        "pages": pages,
        "valid_pages": valid_pages,
        "mask_digits": MASK_DIGITS,
        "latency_ms": {
            "preprocess": round(t_preprocess * 1000, 1),
            "inference": round(t_inference * 1000, 1),
            "postprocess": round(t_postprocess * 1000, 1),
            "ocr_fallback": round(t_ocr_fallback * 1000, 1),
            "s3_save": save_ms,
            "total": total_ms,
        },
    }
    if source_key is not None:
        out["source_key"] = source_key
    if mask_output_key is not None:
        out["mask_output_key"] = mask_output_key

    _write_metadata(source_key, event.get("s3_bucket"), out)
    return out
