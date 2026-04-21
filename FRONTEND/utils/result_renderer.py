import io
import json

import streamlit as st
from PIL import Image

DOC_BADGE = {
    "AADHAAR":  {"emoji": "🪪", "color": "#1E5EBB", "bg": "#E8F0FB"},
    "PAN":      {"emoji": "💳", "color": "#0A7C59", "bg": "#E6F4EF"},
    "PASSPORT": {"emoji": "📘", "color": "#6B3FA0", "bg": "#F0EAF9"},
    "VOTER":    {"emoji": "🗳️", "color": "#B85C00", "bg": "#FEF3E6"},
    "E_VOTER":  {"emoji": "🗳️", "color": "#B85C00", "bg": "#FEF3E6"},
    "GENERIC":  {"emoji": "📄", "color": "#4A5568", "bg": "#F7F8FA"},
}

FIELD_LABELS = {
    "aadhaar": "Aadhaar Number",
    "vid": "VID",
    "name": "Name",
    "dob": "Date of Birth",
    "yob": "Year of Birth",
    "gender": "Gender",
    "father": "Father",
    "mother": "Mother",
    "husband": "Husband",
    "address": "Address",
    "pin": "PIN Code",
    "phone": "Phone",
    "pan": "PAN Number",
    "doi": "Date of Issue",
    "doe": "Date of Expiry",
    "passportNumber": "Passport Number",
    "givenName": "Given Name",
    "surname": "Surname",
    "nationality": "Nationality",
    "placeOfBirth": "Place of Birth",
    "placeOfIssue": "Place of Issue",
    "countryCode": "Country Code",
    "type": "Type",
    "mrzLine1": "MRZ Line 1",
    "mrzLine2": "MRZ Line 2",
    "voterId": "Voter ID",
    "relationName": "Relation Name",
    "age": "Age",
    "ageAsPerDate": "Age (as of date)",
}


def _badge_html(doc_type: str, sub_type: str) -> str:
    info = DOC_BADGE.get(doc_type, DOC_BADGE["GENERIC"])
    label = doc_type + (f" · {sub_type}" if sub_type else "")
    return (
        f'<span style="background:{info["bg"]};color:{info["color"]};'
        f'font-weight:700;font-size:13px;padding:4px 12px;border-radius:20px;'
        f'border:1px solid {info["color"]}33;">'
        f'{info["emoji"]} {label}</span>'
    )


def _verhoeff_chip(passed: bool) -> str:
    if passed:
        return '<span style="background:#E6F4EF;color:#0A7C59;font-size:12px;padding:3px 10px;border-radius:12px;font-weight:600;">✔ Verhoeff Pass</span>'
    return '<span style="background:#FEE9E9;color:#C0392B;font-size:12px;padding:3px 10px;border-radius:12px;font-weight:600;">✖ Verhoeff Fail</span>'


def _mask_status_chip(output_masked: bool) -> str:
    if output_masked:
        return '<span style="background:#E8F0FB;color:#1E5EBB;font-size:12px;padding:3px 10px;border-radius:12px;font-weight:600;">🛡 Output Masked</span>'
    return '<span style="background:#F7F8FA;color:#7A8499;font-size:12px;padding:3px 10px;border-radius:12px;font-weight:600;">⬜ Not Masked</span>'


def _show_masked_preview(masked_bytes: bytes, page_no: int = 1) -> None:
    """Show masked raster with Pillow; masked PDF rasterizes the page matching ``page_no`` (1-based)."""
    if masked_bytes[:4] == b"%PDF":
        import fitz

        doc = fitz.open(stream=masked_bytes, filetype="pdf")
        try:
            n = len(doc)
            if n < 1:
                raise ValueError("empty pdf")
            idx = max(0, min(int(page_no) - 1, n - 1))
            pix = doc[idx].get_pixmap(dpi=144, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        finally:
            doc.close()
        st.image(img, use_container_width=True)
        return
    st.image(Image.open(io.BytesIO(masked_bytes)), use_container_width=True)


def render_document_card(
    doc: dict,
    original_image_bytes: bytes,
    masked_pages: dict[int, bytes] | None,
    file_label: str,
    doc_index: int,
    show_images: bool = False,
    page_images: dict | None = None,
):
    doc_type = doc.get("documentType", "GENERIC").upper()
    sub_type = doc.get("subType", "")
    page_no = doc.get("pageNo", 1)
    ocr_data = doc.get("ocrData", {})
    additional = doc.get("additionalDetails", {})
    masked_bytes = (masked_pages or {}).get(page_no)

    verhoeff = additional.get("verhoeffCheck")
    output_masked = additional.get("outputMaskStatus", False)

    # --- Header row: badge + chips on left, page number on right ---
    left_col, right_col = st.columns([6, 1])
    with left_col:
        chips = _badge_html(doc_type, sub_type)
        if verhoeff is not None:
            chips += "&nbsp;&nbsp;" + _verhoeff_chip(verhoeff)
        if doc_type == "AADHAAR":
            chips += "&nbsp;&nbsp;" + _mask_status_chip(output_masked)
        st.markdown(chips, unsafe_allow_html=True)
    with right_col:
        st.markdown(
            f'<div style="text-align:right;color:#7A8499;font-size:13px;padding-top:4px;">Page {page_no}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div style="color:#4A5568;font-size:12px;margin:4px 0 12px 0;">📁 {file_label}</div>',
        unsafe_allow_html=True,
    )

    # --- OCR Fields ---
    fields = {k: v.get("value", "") for k, v in ocr_data.items() if v.get("value", "")}
    if fields:
        cols = st.columns(2)
        for i, (key, val) in enumerate(fields.items()):
            label = FIELD_LABELS.get(key, key.replace("_", " ").title())
            with cols[i % 2]:
                st.markdown(
                    f'<div style="margin-bottom:10px;">'
                    f'<div style="font-size:11px;color:#7A8499;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;">{label}</div>'
                    f'<div style="font-size:14px;color:#0D1B2A;font-weight:500;">{val}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.markdown('<div style="color:#7A8499;font-style:italic;font-size:13px;">No fields extracted</div>', unsafe_allow_html=True)

    # --- Mask status row (Aadhaar only) ---
    if doc_type == "AADHAAR":
        input_mask = additional.get("inputMaskStatus", {})
        is_already_masked = input_mask.get("isMasked", False)
        if is_already_masked:
            st.markdown(
                '<div style="margin-top:6px;font-size:12px;color:#B85C00;">⚠ Document was already masked on input</div>',
                unsafe_allow_html=True,
            )

    # --- Image section (shown once per file for all doc types) ---
    if show_images and original_image_bytes:
        st.markdown(
            '<div style="margin-top:16px;margin-bottom:8px;font-size:13px;font-weight:600;color:#0D1B2A;">Document Image</div>',
            unsafe_allow_html=True,
        )

        # Prefer the per-page image saved by Lambda (works for PDFs);
        # fall back to the raw uploaded file bytes (works for single images).
        display_bytes = (
            (page_images or {}).get(page_no) or original_image_bytes
        )

        if doc_type == "AADHAAR":
            # Side-by-side: original + masked
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.markdown('<div style="font-size:12px;color:#7A8499;margin-bottom:4px;">Original</div>', unsafe_allow_html=True)
                try:
                    st.image(Image.open(io.BytesIO(display_bytes)), width="stretch")
                except Exception:
                    st.caption("Preview not available")
            with img_col2:
                st.markdown('<div style="font-size:12px;color:#7A8499;margin-bottom:4px;">Masked</div>', unsafe_allow_html=True)
                if masked_bytes:
                    try:
                        _show_masked_preview(masked_bytes, page_no)
                    except Exception:
                        st.markdown(
                            '<div style="border:1px dashed #CBD5E0;border-radius:8px;height:160px;'
                            'display:flex;align-items:center;justify-content:center;'
                            'color:#A0AEC0;font-size:12px;">Masked image unavailable</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        '<div style="border:1px dashed #CBD5E0;border-radius:8px;height:160px;'
                        'display:flex;align-items:center;justify-content:center;'
                        'color:#A0AEC0;font-size:12px;">Awaiting masking…</div>',
                        unsafe_allow_html=True,
                    )
        else:
            # All other doc types: just show the original
            try:
                img_col, _ = st.columns([2, 1])
                with img_col:
                    st.image(Image.open(io.BytesIO(display_bytes)), width="stretch")
            except Exception:
                st.caption("Preview not available")


_render_counter = 0

def render_full_result(
    result_json: dict,
    original_image_bytes: bytes,
    masked_pages: dict[int, bytes] | None,
    file_label: str,
    page_images: dict | None = None,
):
    global _render_counter
    _render_counter += 1
    _uid = _render_counter
    """Render all documents from a single file's pipeline result."""
    docs = result_json.get("result", {}).get("documents", [])

    if not docs:
        st.warning("No documents extracted from this file.")
        return

    # Track which pages have already shown their image preview
    pages_shown: set[int] = set()

    tab_results, tab_json = st.tabs(["📋  Results", "{ }  View JSON"])

    with tab_results:
        for i, doc in enumerate(docs):
            page_no = doc.get("pageNo", 1)
            show_images = page_no not in pages_shown
            if show_images:
                pages_shown.add(page_no)

            if i > 0:
                st.markdown(
                    '<hr style="border:none;border-top:2px solid #1E5EBB;margin:24px 0;">',
                    unsafe_allow_html=True,
                )

            render_document_card(
                doc, original_image_bytes, masked_pages, file_label, i,
                show_images=show_images, page_images=page_images,
            )

    with tab_json:
        # Group documents by documentType for separate JSON views
        from collections import defaultdict
        by_type: dict[str, list] = defaultdict(list)
        for doc in docs:
            by_type[doc.get("documentType", "GENERIC")].append(doc)

        if len(by_type) > 1:
            # Multi-type result (e.g. Aadhaar + PAN in same PDF) — show one section per type
            for doc_type, type_docs in by_type.items():
                badge = DOC_BADGE.get(doc_type, DOC_BADGE["GENERIC"])
                st.markdown(
                    f'<div style="margin:12px 0 6px 0;font-size:13px;font-weight:700;color:{badge["color"]};">'
                    f'{badge["emoji"]} {doc_type}</div>',
                    unsafe_allow_html=True,
                )
                type_result = {
                    "requestId": result_json.get("requestId", ""),
                    "result": {"documents": type_docs},
                    "statusCode": result_json.get("statusCode", 101),
                }
                type_json_str = json.dumps(type_result, indent=2, ensure_ascii=False)
                st.code(type_json_str, language="json")
                st.download_button(
                    label=f"⬇ Download {doc_type} JSON",
                    data=type_json_str.encode(),
                    file_name=f"{file_label}_{doc_type}_result.json",
                    mime="application/json",
                    key=f"dl_json_{file_label}_{doc_type}_{_uid}",
                )
        else:
            # Single doc type — show the full JSON as usual
            json_str = json.dumps(result_json, indent=2, ensure_ascii=False)
            st.code(json_str, language="json")
            st.download_button(
                label="⬇ Download JSON",
                data=json_str.encode(),
                file_name=f"{file_label}_result.json",
                mime="application/json",
                key=f"dl_json_{file_label}_{_uid}",
            )
