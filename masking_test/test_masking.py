#!/usr/bin/env python3
"""
Local test for the Aadhaar masking Lambda.

Runs YOLO ONNX inference on input images/PDFs, detects Aadhaar card elements,
validates front/back, masks QR codes and Aadhaar numbers, and saves results.

Usage:
  cd masking_test
  ./venv/bin/python test_masking.py --image "../Test samples/Sample_1.jpg"
  ./venv/bin/python test_masking.py --image "/path/to/aadhaar.pdf"
  ./venv/bin/python test_masking.py --dir "../Test samples/" --batch
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

os.environ["VALIDATE_DEBUG"] = "1"
os.environ["MODEL_PATH"] = str(Path(__file__).parent / "best_new.onnx")
os.environ["CONF"] = "0.40"
os.environ.setdefault("MASK_DIGITS", "8")

from handler import (
    preprocess,
    postprocess,
    postprocess_low_conf,
    expand_mask_boxes_for_digits,
    get_session,
    validate_from_detections,
    apply_mask_pillow,
    ocr_fallback_mask,
    _letterbox_to_orig,
    CLASS_NAMES,
    MASK_CLASSES,
    CONF_THRESHOLD,
    MODEL_CONF,
    IOU_THRESH,
    MAX_DET,
    AGNOSTIC_NMS,
    IMGSZ,
    OCR_FALLBACK,
    MASK_DIGITS,
)

FRONT_ONLY_CLASSES = {"aadhaar_photo", "aadhaar_dob", "aadhaar_gender", "aadhaar_holder_name"}
BACK_ONLY_CLASSES = {"aadhaar_address"}
BOTH_SIDES_CLASSES = {"aadhaar_no", "aadhar_no_mask", "aadhaar_qr", "aadhaar_logo", "emblem", "gov_logo"}


def _get_side(cls_name: str) -> str:
    if cls_name in FRONT_ONLY_CLASSES:
        return "front"
    if cls_name in BACK_ONLY_CLASSES:
        return "back"
    return "both"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("masking_test")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "Test Output"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
PDF_EXTS = {".pdf"}


def load_images(file_path: Path) -> tuple[bool, list[np.ndarray]]:
    """Load image(s) from file. Returns (is_pdf, list_of_rgb_arrays)."""
    if file_path.suffix.lower() in PDF_EXTS:
        import fitz
        doc = fitz.open(str(file_path))
        images = []
        dpi = int(os.environ.get("PDF_DPI", "150"))
        for i in range(len(doc)):
            page = doc[i]
            pix = page.get_pixmap(dpi=dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = img[:, :, :3].copy()
            elif pix.n == 1:
                gray = np.squeeze(img)
                img = np.stack([gray, gray, gray], axis=-1)
            images.append(img)
        doc.close()
        return True, images
    else:
        pil = Image.open(str(file_path)).convert("RGB")
        return False, [np.array(pil)]


def process_single(file_path: Path, session) -> dict:
    """Process a single image/PDF through the masking pipeline."""
    input_name = session.get_inputs()[0].name
    is_pdf, images = load_images(file_path)
    log.info("Loaded %s: %d page(s), is_pdf=%s", file_path.name, len(images), is_pdf)

    page_results = []
    t_total_start = time.time()

    for page_idx, img in enumerate(images):
        page_num = page_idx + 1
        h, w = img.shape[:2]
        log.info("Page %d/%d: %dx%d", page_num, len(images), w, h)

        t_pp = time.time()
        x, scale, (pad_x, pad_y, orig_w, orig_h) = preprocess(img)
        pp_ms = round((time.time() - t_pp) * 1000, 1)

        t_inf = time.time()
        outputs = session.run(None, {input_name: x})
        inf_ms = round((time.time() - t_inf) * 1000, 1)

        t_post = time.time()
        detected, mask_boxes, all_detections = postprocess(
            outputs[0], MODEL_CONF, IOU_THRESH, MAX_DET, agnostic_nms=AGNOSTIC_NMS
        )
        mask_boxes = expand_mask_boxes_for_digits(
            mask_boxes, MASK_DIGITS, img_width=float(IMGSZ),
        )
        post_ms = round((time.time() - t_post) * 1000, 1)

        low_conf_detections = postprocess_low_conf(outputs[0], low_thresh=0.15)

        validation = validate_from_detections(
            detected, CONF_THRESHOLD,
            source_label=f"{file_path.name} p.{page_num}",
        )

        masked_cls = sorted({name for *_, name in mask_boxes})
        validation["masked_classes"] = masked_cls

        boxes_orig = []
        if mask_boxes:
            boxes_orig = [
                _letterbox_to_orig(x1, y1, x2, y2, scale, pad_x, pad_y)
                for (x1, y1, x2, y2, *_rest) in mask_boxes
            ]

        # OCR fallback: catch Aadhaar numbers YOLO missed
        ocr_fallback_boxes = []
        t_fb = time.time()
        if OCR_FALLBACK:
            ocr_fallback_boxes = ocr_fallback_mask(img, boxes_orig)
            if ocr_fallback_boxes:
                boxes_orig.extend(ocr_fallback_boxes)
                validation["ocr_fallback_count"] = len(ocr_fallback_boxes)
        fb_ms = round((time.time() - t_fb) * 1000, 1)

        log.info("Page %d latency: preprocess=%.1fms, inference=%.1fms, postprocess=%.1fms, ocr_fallback=%.1fms",
                 page_num, pp_ms, inf_ms, post_ms, fb_ms)

        page_results.append({
            "page_num": page_num,
            "img": img,
            "detected": sorted(detected),
            "all_detections": all_detections,
            "low_conf_detections": low_conf_detections,
            "mask_boxes": mask_boxes,
            "boxes_orig": boxes_orig,
            "ocr_fallback_boxes": ocr_fallback_boxes,
            "validation": validation,
            "scale": scale,
            "pad_x": pad_x,
            "pad_y": pad_y,
            "latency_ms": {"preprocess": pp_ms, "inference": inf_ms, "postprocess": post_ms, "ocr_fallback": fb_ms},
        })

    OUTPUT_DIR.mkdir(exist_ok=True)

    for pr in page_results:
        page_num = pr["page_num"]
        val = pr["validation"]
        detected = pr["detected"]
        all_dets = pr["all_detections"]
        low_dets = pr["low_conf_detections"]
        mask_boxes = pr["mask_boxes"]
        boxes_orig = pr["boxes_orig"]

        print(f"\n{'='*60}")
        print(f"MASKING RESULT — {file_path.name} (Page {page_num}/{len(images)})")
        print(f"{'='*60}")
        print(f"  Card type       : {val['card_type']}")
        print(f"  Validation      : {val['validation']} (score: {val['score']})")
        print(f"  Conf threshold  : {CONF_THRESHOLD}")
        print(f"  Mask digits     : {MASK_DIGITS}/12")
        print(f"  Missing required: {val.get('missing_required', [])}")
        pg_lat = pr.get("latency_ms", {})
        print(f"  Latency         : preprocess={pg_lat.get('preprocess',0)}ms, inference={pg_lat.get('inference',0)}ms, "
              f"postprocess={pg_lat.get('postprocess',0)}ms, ocr_fallback={pg_lat.get('ocr_fallback',0)}ms")
        print(f"  Reason          : {val['reason']}")

        print(f"\n  DETECTIONS ABOVE THRESHOLD (conf >= {MODEL_CONF}):")
        print(f"  {'Class':<25s} {'Confidence':>10s}  {'Side':<6s}  {'Masked?':>7s}")
        print(f"  {'-'*25} {'-'*10}  {'-'*6}  {'-'*7}")
        for d in sorted(all_dets, key=lambda x: -x["confidence"]):
            masked_flag = "YES" if d["masked"] else ""
            side = _get_side(d["class"])
            print(f"  {d['class']:<25s} {d['confidence']:>10.4f}  {side:<6s}  {masked_flag:>7s}")

        mask_related = [d for d in low_dets if d["class"] in ("aadhar_no_mask", "aadhaar_no", "aadhaar_qr")]
        below_thresh = [d for d in mask_related if d["confidence"] < MODEL_CONF]
        if below_thresh:
            print(f"\n  WARNING: MASK-RELEVANT DETECTIONS BELOW THRESHOLD (conf < {MODEL_CONF}):")
            for d in below_thresh:
                side = _get_side(d["class"])
                print(f"    {d['class']:<25s} {d['confidence']:>10.4f}  {side:<6s}  FILTERED OUT")

        ocr_fb = pr.get("ocr_fallback_boxes", [])
        yolo_count = len(mask_boxes)
        total_mask = yolo_count + len(ocr_fb)
        print(f"\n  MASK REGIONS: {total_mask} (YOLO: {yolo_count}, OCR fallback: {len(ocr_fb)})")
        for i, box in enumerate(mask_boxes):
            cls_name = box[4]
            conf = box[5]
            orig = boxes_orig[i] if i < len(boxes_orig) else ("?",)*4
            print(f"    [{i+1}] {cls_name} (conf={conf:.4f}): orig=({orig[0]},{orig[1]})-({orig[2]},{orig[3]})")
        for i, box in enumerate(ocr_fb):
            idx = yolo_count + i + 1
            print(f"    [{idx}] OCR_FALLBACK_AADHAAR (ocr): ({box[0]},{box[1]})-({box[2]},{box[3]})")

        print(f"{'='*60}")

        if boxes_orig:
            jpeg_bytes = apply_mask_pillow(pr["img"], boxes_orig)
            suffix = f"_p{page_num}" if len(images) > 1 else ""
            out_path = OUTPUT_DIR / f"{file_path.stem}{suffix}_masked.jpg"
            out_path.write_bytes(jpeg_bytes)
            log.info("Masked image saved → %s", out_path)
        else:
            log.info("Page %d: no regions to mask (card_type=%s, validation=%s)",
                     page_num, val["card_type"], val["validation"])

    masking_total_sec = round(time.time() - t_total_start, 1)
    log.info("Total masking time: %.1fs (%d pages)", masking_total_sec, len(images))

    summary = {
        "file": file_path.name,
        "total_pages": len(images),
        "conf_threshold": CONF_THRESHOLD,
        "model_conf": MODEL_CONF,
        "mask_digits": MASK_DIGITS,
        "masking_total_sec": masking_total_sec,
        "pages": [
            {
                "page": pr["page_num"],
                "card_type": pr["validation"]["card_type"],
                "validation": pr["validation"]["validation"],
                "score": pr["validation"]["score"],
                "detections": [
                    {"class": d["class"], "confidence": d["confidence"], "side": _get_side(d["class"]), "masked": d["masked"]}
                    for d in sorted(pr["all_detections"], key=lambda x: -x["confidence"])
                ],
                "below_threshold_mask_classes": [
                    {"class": d["class"], "confidence": d["confidence"], "side": _get_side(d["class"])}
                    for d in pr["low_conf_detections"]
                    if d["class"] in ("aadhar_no_mask", "aadhaar_no", "aadhaar_qr")
                    and d["confidence"] < MODEL_CONF
                ],
                "missing_required": pr["validation"].get("missing_required", []),
                "masked_classes": pr["validation"].get("masked_classes", []),
                "mask_regions_yolo": len(pr["mask_boxes"]),
                "mask_regions_ocr_fallback": len(pr.get("ocr_fallback_boxes", [])),
                "mask_regions_total": len(pr["boxes_orig"]),
                "ocr_fallback_boxes": [
                    {"box": list(b), "source": "ocr_fallback"}
                    for b in pr.get("ocr_fallback_boxes", [])
                ],
                "reason": pr["validation"]["reason"],
                "latency_ms": pr.get("latency_ms", {}),
            }
            for pr in page_results
        ],
    }

    json_path = OUTPUT_DIR / f"{file_path.stem}_masking.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Masking JSON saved → %s", json_path)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Test Aadhaar masking Lambda locally")
    parser.add_argument("--image", help="Path to a single input image or PDF")
    parser.add_argument("--dir", help="Path to directory of images (use with --batch)")
    parser.add_argument("--batch", action="store_true", help="Process all supported files in --dir")
    parser.add_argument(
        "--mask-digits",
        type=int,
        default=8,
        choices=[8, 12],
        help="How many Aadhaar digits to mask (8=last 4 visible, 12=fully masked, default: 8)",
    )
    args = parser.parse_args()

    os.environ["MASK_DIGITS"] = str(args.mask_digits)

    if not args.image and not (args.dir and args.batch):
        parser.error("Provide --image or --dir with --batch")

    model_path = os.environ["MODEL_PATH"]
    log.info("Loading YOLO model: %s", model_path)
    session = get_session(model_path)
    log.info("Model loaded. Input: %s", session.get_inputs()[0].shape)

    supported = IMAGE_EXTS | PDF_EXTS

    if args.batch and args.dir:
        input_dir = Path(args.dir)
        if not input_dir.is_dir():
            log.error("Not a directory: %s", input_dir)
            sys.exit(1)

        files = sorted(f for f in input_dir.iterdir() if f.suffix.lower() in supported)
        log.info("Batch mode: %d files in %s", len(files), input_dir)

        results = []
        for i, fpath in enumerate(files, 1):
            log.info("\n>>> [%d/%d] %s", i, len(files), fpath.name)
            summary = process_single(fpath, session)
            results.append(summary)

        print(f"\n{'='*60}")
        print("BATCH SUMMARY")
        print(f"{'='*60}")
        for r in results:
            for pg in r["pages"]:
                status = f"{pg['validation']} ({pg['score']})"
                masked = f"{pg['mask_regions']} regions" if pg["mask_regions"] > 0 else "none"
                print(f"  {r['file']:30s} p{pg['page']} → {pg['card_type']:6s} {status:20s} masked: {masked}")
        print(f"{'='*60}")
    else:
        file_path = Path(args.image)
        if not file_path.exists():
            log.error("File not found: %s", file_path)
            sys.exit(1)
        process_single(file_path, session)


if __name__ == "__main__":
    main()
