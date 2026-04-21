"""
OCR Engine for Lambda — PaddleOCR ONNX via RapidOCR.

Adapted from code/ocr_extract.py. Models are loaded from S3 instead of
HuggingFace and cached in /tmp/models/ across warm Lambda invocations.
"""
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from rapidocr_onnxruntime import RapidOCR

log = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_DIR = Path("/tmp/models")
PDF_DPI = 250
_DEFAULT_PDF_OCR_MAX_WORKERS = 4
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
PDF_EXTS = {".pdf"}

# S3 keys for the three model files (relative to MODEL_PREFIX)
_MODEL_FILES = {
    "det": "det.onnx",
    "rec": "rec.onnx",
    "dict": "dict.txt",
}

# Module-level cache (persists across warm invocations)
_engine: RapidOCR | None = None


# ── Model Download ───────────────────────────────────────────────────────────

def _download_models_from_s3(s3_client, bucket: str, prefix: str) -> dict[str, str]:
    """Download PaddleOCR models from S3 to /tmp/models/ if not already cached."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}

    for key, filename in _MODEL_FILES.items():
        local_path = MODEL_DIR / filename
        if local_path.exists():
            log.info("Model cached: %s", local_path)
            paths[key] = str(local_path)
            continue

        s3_key = f"{prefix}/{filename}"
        log.info("Downloading s3://%s/%s → %s", bucket, s3_key, local_path)
        s3_client.download_file(bucket, s3_key, str(local_path))
        paths[key] = str(local_path)

    return paths


def _build_engine(paths: dict[str, str]) -> RapidOCR:
    """Initialise the RapidOCR engine with model paths."""
    import yaml
    import rapidocr_onnxruntime

    pkg_dir = Path(rapidocr_onnxruntime.__file__).parent
    cfg_path = pkg_dir / "config.yaml"
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    config["Det"]["model_path"] = paths["det"]
    config["Rec"]["model_path"] = paths["rec"]
    # Newer rapidocr uses "rec_keys_path"; older uses "keys_path"
    config["Rec"]["rec_keys_path"] = paths["dict"]

    cfg_out = "/tmp/rapidocr_config.yaml"
    with open(cfg_out, "w") as f:
        yaml.dump(config, f)

    return RapidOCR(config_path=cfg_out)


def get_engine(s3_client, bucket: str, prefix: str) -> RapidOCR:
    """Return the cached OCR engine, downloading models on cold start."""
    global _engine
    if _engine is not None:
        return _engine

    t0 = time.time()
    paths = _download_models_from_s3(s3_client, bucket, prefix)
    _engine = _build_engine(paths)
    log.info("OCR engine ready (%.1fs)", time.time() - t0)
    return _engine


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pil_to_jpeg_bytes(img: Image.Image) -> bytes:
    """Convert a PIL Image to JPEG bytes."""
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _pdf_ocr_max_workers(num_pages: int) -> int:
    """Cap concurrent page OCR workers (configurable via env)."""
    raw = os.environ.get("PDF_OCR_MAX_WORKERS", str(_DEFAULT_PDF_OCR_MAX_WORKERS))
    try:
        cap = max(1, int(raw.strip()))
    except ValueError:
        cap = _DEFAULT_PDF_OCR_MAX_WORKERS
    return min(num_pages, cap)


def _pdf_page_worker(
    args: tuple[RapidOCR, int, "Image.Image"],
) -> tuple[int, list, str, bytes, int, int, int, int, int]:
    """OCR one PDF page in a parallel worker; returns per-page metrics."""
    engine, page_num, img = args
    arr = np.array(img)
    h, w = arr.shape[:2]

    t_ocr = time.time()
    blocks = ocr_image(engine, img)
    page_ocr_ms = int((time.time() - t_ocr) * 1000)

    page_text = "\n".join(b["text"] for b in blocks)

    t_jpg = time.time()
    jpeg_bytes = _pil_to_jpeg_bytes(img)
    jpg_ms = int((time.time() - t_jpg) * 1000)

    return page_num, blocks, page_text, jpeg_bytes, w, h, page_ocr_ms, len(blocks), jpg_ms


# ── OCR Logic ────────────────────────────────────────────────────────────────

def ocr_image(engine: RapidOCR, img_input) -> list[dict]:
    """
    Run OCR on an image (file path or PIL.Image) and return structured blocks.

    RapidOCR result item format:
        [ [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], "text", 0.98 ]
    """
    t_prep = time.time()
    if isinstance(img_input, Image.Image):
        img_input = ImageOps.exif_transpose(img_input)
        arr = np.array(img_input.convert("RGB"))
    else:
        pil = ImageOps.exif_transpose(Image.open(img_input))
        arr = np.array(pil.convert("RGB"))
    prep_ms = int((time.time() - t_prep) * 1000)

    t_infer = time.time()
    result, _ = engine(arr)
    infer_ms = int((time.time() - t_infer) * 1000)

    log.info("[LATENCY] ocr_detail: img_prep=%dms, inference=%dms", prep_ms, infer_ms)

    if not result:
        return []

    blocks = []
    for bbox, text, conf in result:
        blocks.append({
            "text": text,
            "confidence": round(float(conf), 4),
            "bounding_box": [
                [round(float(x), 1), round(float(y), 1)] for x, y in bbox
            ],
        })

    if not blocks:
        log.warning("OCR: zero blocks detected — image may be blank or unreadable")

    return blocks


def process_image(engine: RapidOCR, path: Path) -> dict:
    """Process a single image file."""
    log.info("Image: %s", path.name)

    cv_img = cv2.imread(str(path))
    if cv_img is not None:
        h, w = cv_img.shape[:2]
        log.info("[LATENCY] image_resolution=%dx%d (%s)", w, h, path.name)

    t0 = time.time()
    blocks = ocr_image(engine, path)
    ocr_ms = int((time.time() - t0) * 1000)
    log.info("[LATENCY] image_ocr=%dms (%d blocks)", ocr_ms, len(blocks))

    full_text = "\n".join(b["text"] for b in blocks)
    image_bytes = path.read_bytes()

    return {
        "file": path.name,
        "type": "image",
        "pages": [{
            "page": 1,
            "blocks": blocks,
            "text": full_text,
        }],
        "full_text": full_text,
        "total_blocks": len(blocks),
        "elapsed_seconds": round(time.time() - t0, 3),
        "image_bytes_list": [image_bytes],
    }


def process_pdf(engine: RapidOCR, path: Path, dpi: int = PDF_DPI) -> dict:
    """Process a PDF file, OCR'ing each page at the given DPI (parallel)."""
    from pdf2image import convert_from_path

    log.info("PDF: %s (dpi=%d)", path.name, dpi)
    t_total = time.time()

    t_convert = time.time()
    images = convert_from_path(str(path), dpi=dpi)
    convert_ms = int((time.time() - t_convert) * 1000)
    n = len(images)
    log.info("[LATENCY] pdf_to_images=%dms (%d pages)", convert_ms, n)

    if n == 0:
        elapsed = round(time.time() - t_total, 3)
        return {
            "file": path.name, "type": "pdf", "total_pages": 0,
            "pages": [], "full_text": "", "total_blocks": 0,
            "elapsed_seconds": elapsed, "image_bytes_list": [],
            "dpi": dpi, "convert_ms": convert_ms,
        }

    max_workers = _pdf_ocr_max_workers(n)
    log.info(
        "Parallel OCR: ThreadPoolExecutor (max_workers=%d, pages=%d)",
        max_workers, n,
    )

    worker_items = [(engine, i, img) for i, img in enumerate(images, 1)]

    pages = []
    all_text = []
    all_image_bytes = []
    total_blocks = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for (page_num, blocks, page_text, jpeg_bytes,
             w, h, page_ocr_ms, block_count, jpg_ms) in pool.map(_pdf_page_worker, worker_items):
            log.info("[LATENCY] page_%d: resolution=%dx%d", page_num, w, h)
            log.info(
                "[LATENCY] page_%d: ocr=%dms (%d blocks), jpeg_convert=%dms",
                page_num, page_ocr_ms, block_count, jpg_ms,
            )
            pages.append({"page": page_num, "blocks": blocks, "text": page_text})
            all_text.append(page_text)
            all_image_bytes.append(jpeg_bytes)
            total_blocks += block_count

    elapsed = round(time.time() - t_total, 3)
    log.info(
        "[LATENCY] pdf_ocr_total=%dms (%d blocks, %d pages)",
        int(elapsed * 1000), total_blocks, n,
    )

    return {
        "file": path.name,
        "type": "pdf",
        "total_pages": n,
        "pages": pages,
        "full_text": "\n\n".join(all_text),
        "total_blocks": total_blocks,
        "elapsed_seconds": elapsed,
        "image_bytes_list": all_image_bytes,
        "dpi": dpi,
        "convert_ms": convert_ms,
    }
