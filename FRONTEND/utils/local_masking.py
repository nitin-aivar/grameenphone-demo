"""
Local YOLO-based Aadhaar masking — runs entirely on the host machine.
Used by the Streamlit app so masking respects the mask_digits parameter
without requiring redeployment of the AWS Lambda.
"""
from __future__ import annotations

import io
import os
import sys
import logging
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MASKING_DIR = _REPO_ROOT / "masking_test"
_MODEL_PATH = _MASKING_DIR / "best_new.onnx"

if str(_MASKING_DIR) not in sys.path:
    sys.path.insert(0, str(_MASKING_DIR))

os.environ.setdefault("MODEL_PATH", str(_MODEL_PATH))
os.environ.setdefault("CONF", "0.40")
os.environ.setdefault("MASK_DIGITS", "8")

_session = None


def _get_session():
    global _session
    if _session is None:
        from handler import get_session
        _session = get_session(str(_MODEL_PATH))
    return _session


def mask_image_locally(file_bytes: bytes, filename: str, mask_digits: int = 8) -> bytes | None:
    """
    Run YOLO masking on a single image/PDF page locally.
    Returns JPEG bytes of the masked image, or None if no masking needed.
    """
    os.environ["MASK_DIGITS"] = str(mask_digits)

    import handler as _h
    _h.MASK_DIGITS = mask_digits

    from handler import (
        preprocess, postprocess, apply_mask_pillow,
        ocr_fallback_mask, expand_mask_boxes_for_digits,
        _letterbox_to_orig,
        MODEL_CONF, IOU_THRESH, MAX_DET, AGNOSTIC_NMS, IMGSZ, OCR_FALLBACK,
    )

    session = _get_session()
    input_name = session.get_inputs()[0].name

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "pdf":
        try:
            import fitz
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            images = []
            for page in doc:
                pix = page.get_pixmap(dpi=150)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                if pix.n == 4:
                    arr = arr[:, :, :3]
                images.append(arr)
            doc.close()
        except Exception as exc:
            log.warning("Local PDF masking failed: %s", exc)
            return None
    else:
        pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        images = [np.array(pil)]

    any_masked = False
    for img in images:
        x, scale, (pad_x, pad_y, _ow, _oh) = preprocess(img)
        outputs = session.run(None, {input_name: x})

        _, mask_boxes, _ = postprocess(
            outputs[0], MODEL_CONF, IOU_THRESH, MAX_DET, agnostic_nms=AGNOSTIC_NMS
        )
        mask_boxes = expand_mask_boxes_for_digits(
            mask_boxes, mask_digits, img_width=float(IMGSZ),
        )

        boxes_orig = [
            _letterbox_to_orig(bx1, by1, bx2, by2, scale, pad_x, pad_y)
            for (bx1, by1, bx2, by2, *_rest) in mask_boxes
        ] if mask_boxes else []

        if OCR_FALLBACK:
            ocr_extra = ocr_fallback_mask(img, boxes_orig)
            if ocr_extra:
                boxes_orig.extend(ocr_extra)

        if boxes_orig:
            any_masked = True
            masked_bytes = apply_mask_pillow(img, boxes_orig)
            return masked_bytes

    return None if not any_masked else None
