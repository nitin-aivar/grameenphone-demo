#!/usr/bin/env python3
"""
Local test for the Lambda pipeline — runs OCR + Mistral vision without S3.

Uses HuggingFace for models (like the local code/ocr_extract.py) and
calls Bedrock directly. This validates the core logic before deployment.

Usage:
  cd lambda
  python test_local.py --profile my-sso-profile --image ../code/input/Aadhar1.png
  python test_local.py --profile my-sso-profile --image ../code/input/PAN1.png
  python test_local.py --profile my-sso-profile --image ../code/input/Aadhar1.png --ocr-only
  python test_local.py --profile my-sso-profile --dir ../code/input/ --batch
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test")

# ── Local model loader (HuggingFace, same as code/ocr_extract.py) ────────

def build_engine_from_hf():
    """Download models from HuggingFace and build OCR engine."""
    import yaml
    from huggingface_hub import hf_hub_download
    from rapidocr_onnxruntime import RapidOCR
    import rapidocr_onnxruntime

    log.info("Downloading models from HuggingFace …")
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


def process_single(engine, image_path: Path, bedrock_client, model_id: str, ocr_only: bool) -> dict | None:
    """Process a single image/PDF through the pipeline. Returns output dict or None for ocr-only."""
    from ocr_engine import process_image, process_pdf, IMAGE_EXTS, PDF_EXTS
    from doc_classifier import classify_document

    ext = image_path.suffix.lower()
    if ext not in IMAGE_EXTS and ext not in PDF_EXTS:
        log.error("Unsupported file type: %s", ext)
        return None

    # 1. Run OCR
    t0 = time.time()
    if ext in PDF_EXTS:
        ocr_result = process_pdf(engine, image_path)
    else:
        ocr_result = process_image(engine, image_path)

    log.info(
        "OCR: %d blocks in %.1fs",
        ocr_result["total_blocks"],
        time.time() - t0,
    )
    log.info("OCR text preview:\n%s", ocr_result["full_text"][:500])

    # 2. Classify document type via regex
    doc_type = classify_document(ocr_result["full_text"])
    early_aadhaar = doc_type == "aadhaar"
    log.info("Regex classification: %s (early_aadhaar=%s)", doc_type, early_aadhaar)

    # Save OCR result to Test Output/
    test_output_dir = Path(__file__).resolve().parent.parent / "Test Output"
    test_output_dir.mkdir(exist_ok=True)
    ocr_out_path = test_output_dir / f"{image_path.stem}_ocr.json"

    # Compute confidence metrics across all blocks
    all_confidences = [
        b["confidence"]
        for p in ocr_result["pages"]
        for b in p.get("blocks", [])
    ]
    avg_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
    min_conf = min(all_confidences) if all_confidences else 0.0
    low_conf_blocks = sum(1 for c in all_confidences if c < 0.85)

    ocr_dump = {
        "file": image_path.name,
        "doc_type": doc_type,
        "early_aadhaar": early_aadhaar,
        "full_text": ocr_result["full_text"],
        "total_blocks": ocr_result["total_blocks"],
        "elapsed_seconds": ocr_result["elapsed_seconds"],
        "avg_confidence": round(avg_conf, 4),
        "min_confidence": round(min_conf, 4),
        "low_confidence_blocks": low_conf_blocks,
        "pages": [
            {
                "page": p["page"],
                "text": p["text"],
                "blocks": [
                    {"text": b["text"], "confidence": b["confidence"]}
                    for b in p.get("blocks", [])
                ],
            }
            for p in ocr_result["pages"]
        ],
    }
    ocr_out_path.write_text(json.dumps(ocr_dump, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("OCR result saved → %s", ocr_out_path)

    print(f"\n{'='*60}")
    print(f"OCR ACCURACY SUMMARY — {image_path.name}")
    print(f"{'='*60}")
    print(f"  Total blocks      : {ocr_result['total_blocks']}")
    print(f"  Avg confidence    : {avg_conf:.1%}")
    print(f"  Min confidence    : {min_conf:.1%}")
    print(f"  Low conf (<85%)   : {low_conf_blocks} blocks")
    if low_conf_blocks > 0:
        print("\n  Low-confidence blocks:")
        for p in ocr_result["pages"]:
            for b in p.get("blocks", []):
                if b["confidence"] < 0.85:
                    print(f"    [{b['confidence']:.2%}]  {b['text']!r}")
    print(f"{'='*60}\n")

    if ocr_only:
        return None

    # 3. Call Mistral vision via Bedrock
    from json_formatter import format_ocr_result

    log.info("Calling Bedrock (%s) with doc_type=%s …", model_id, doc_type)
    t1 = time.time()
    documents = format_ocr_result(bedrock_client, ocr_result, model_id, doc_type)
    log.info("Bedrock: %d documents in %.1fs", len(documents), time.time() - t1)

    # 4. Build output (same as handler.py)
    has_aadhaar_in_json = any(d.get("documentType") == "AADHAAR" for d in documents)
    output = {
        "requestId": str(uuid.uuid4()),
        "result": {"documents": documents},
        "statusCode": 101,
    }

    # 5. Save final JSON to same Test Output/ folder
    out_path = test_output_dir / f"{image_path.stem}.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=4), encoding="utf-8")

    # 6. Summary
    doc_types = [d.get("documentType", "?") for d in documents]
    log.info("=" * 60)
    log.info("File       : %s", image_path.name)
    log.info("Result     : %s", out_path)
    log.info("Regex type : %s", doc_type)
    log.info("Documents  : %d (%s)", len(documents), ", ".join(doc_types))
    log.info("Aadhaar    : early=%s, json=%s", early_aadhaar, has_aadhaar_in_json)
    if early_aadhaar or has_aadhaar_in_json:
        log.info("  → Would trigger masking Lambda")
    log.info("Total time : %.1fs", time.time() - t0)
    log.info("=" * 60)

    return output


def main():
    parser = argparse.ArgumentParser(description="Test Lambda pipeline locally")
    parser.add_argument("--image", help="Path to a single input image or PDF")
    parser.add_argument("--dir", help="Path to directory of images (use with --batch)")
    parser.add_argument("--batch", action="store_true", help="Process all supported files in --dir")
    parser.add_argument("--profile", default=None, help="AWS CLI profile for Bedrock")
    parser.add_argument(
        "--model",
        default="mistral.ministral-3-14b-instruct",
        help="Bedrock model ID (default: mistral.ministral-3-14b-instruct)",
    )
    parser.add_argument(
        "--region",
        default="ap-south-1",
        help="Bedrock region (default: ap-south-1)",
    )
    parser.add_argument("--ocr-only", action="store_true", help="Run OCR + classifier only, skip Bedrock")
    args = parser.parse_args()

    if not args.image and not (args.dir and args.batch):
        parser.error("Provide --image or --dir with --batch")

    # Build OCR engine once
    engine = build_engine_from_hf()

    # Set up Bedrock client (skip if ocr-only)
    bedrock_client = None
    if not args.ocr_only:
        import boto3
        session_kwargs = {}
        if args.profile:
            session_kwargs["profile_name"] = args.profile
        session = boto3.Session(**session_kwargs)
        bedrock_client = session.client("bedrock-runtime", region_name=args.region)

    from ocr_engine import IMAGE_EXTS, PDF_EXTS
    supported = IMAGE_EXTS | PDF_EXTS

    if args.batch and args.dir:
        # Batch mode: process all supported files in directory
        input_dir = Path(args.dir)
        if not input_dir.is_dir():
            log.error("Not a directory: %s", input_dir)
            sys.exit(1)

        files = sorted(f for f in input_dir.iterdir() if f.suffix.lower() in supported)
        log.info("Batch mode: %d files in %s", len(files), input_dir)

        results = []
        for i, fpath in enumerate(files, 1):
            log.info("\n>>> [%d/%d] %s", i, len(files), fpath.name)
            output = process_single(engine, fpath, bedrock_client, args.model, args.ocr_only)
            if output:
                doc_types = [d.get("documentType", "?") for d in output["result"]["documents"]]
                results.append({"file": fpath.name, "doc_types": doc_types})

        if results:
            log.info("\n" + "=" * 60)
            log.info("BATCH SUMMARY")
            log.info("=" * 60)
            for r in results:
                log.info("  %-30s → %s", r["file"], ", ".join(r["doc_types"]))
    else:
        # Single file mode
        image_path = Path(args.image)
        if not image_path.exists():
            log.error("File not found: %s", image_path)
            sys.exit(1)
        process_single(engine, image_path, bedrock_client, args.model, args.ocr_only)


if __name__ == "__main__":
    main()
