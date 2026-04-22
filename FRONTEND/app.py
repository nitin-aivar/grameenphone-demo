import base64
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from utils.s3_ops import (
    fetch_masked_image,
    fetch_page_image,
    is_aadhaar_result,
    poll_json_result,
    upload_file,
)
from utils.result_renderer import render_full_result

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KYC Intelligent Document Processing",
    page_icon="🪪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }

/* App background */
.stApp { background: #EEF2F7; }

/* Remove default top padding */
.block-container { padding-top: 0 !important; max-width: 1100px; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 2px dashed #B8C8E8;
    border-radius: 12px;
    padding: 8px;
}
[data-testid="stFileUploader"]:hover {
    border-color: #1E5EBB;
}

/* Primary button */
.stButton > button[kind="primary"] {
    background: #1E5EBB !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 10px 32px !important;
    letter-spacing: 0.3px;
}
.stButton > button[kind="primary"]:hover {
    background: #174fa3 !important;
}

/* Download button */
.stDownloadButton > button {
    background: #F0F4FF !important;
    color: #1E5EBB !important;
    border: 1px solid #B8C8E8 !important;
    border-radius: 6px !important;
    font-size: 12px !important;
    padding: 4px 14px !important;
    font-weight: 600 !important;
}

/* Progress / info boxes */
[data-testid="stAlert"] { border-radius: 8px; }

/* Tabs */
[data-testid="stTabs"] button {
    font-weight: 600;
    font-size: 14px;
}

/* Sidebar hide */
[data-testid="collapsedControl"] { display: none; }

/* Divider */
hr { border-color: #E2E8F0; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
_logo_path = Path(__file__).parent / "assets" / "Aivar_v2.png"
if _logo_path.exists():
    _logo_b64 = base64.b64encode(_logo_path.read_bytes()).decode()
    _logo_html = f'<img src="data:image/png;base64,{_logo_b64}" style="height:56px;" />'
else:
    _logo_html = '<div style="font-size:36px;">🪪</div>'

st.markdown(
    f"""
<div style="background:linear-gradient(135deg,#0D1B2A 0%,#1E3A5F 100%);
     padding:28px 40px 22px 40px;margin:-1rem -1rem 28px -1rem;
     border-radius:0 0 16px 16px;">
  <div style="display:flex;align-items:center;gap:14px;">
    {_logo_html}
    <div>
      <div style="color:#FFFFFF;font-size:22px;font-weight:700;letter-spacing:-0.3px;">
        KYC Intelligent Document Processing
      </div>
      <div style="color:#7BA7D4;font-size:13px;margin-top:2px;">
        Powered by Aivar Innovations &nbsp;·&nbsp; Classification for KYC documents and Aadhaar masking
      </div>
    </div>
    <div style="margin-left:auto;text-align:right;">
      <div style="background:#1E5EBB22;border:1px solid #1E5EBB55;border-radius:8px;
           padding:6px 14px;display:inline-block;">
        <span style="color:#7BA7D4;font-size:11px;font-weight:600;text-transform:uppercase;
              letter-spacing:1px;">v1.0 · Demo</span>
      </div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Session state init ────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = []
if "processing" not in st.session_state:
    st.session_state.processing = False


def _restore_results_from_url():
    """Re-fetch results from S3 using stems saved in URL query params."""
    params = st.query_params
    encoded = params.get("r")
    if not encoded or st.session_state.results:
        return
    entries = encoded.split(",")
    restored = []
    for entry in entries:
        if ":" not in entry:
            continue
        stem, filename = entry.split(":", 1)
        result_json = poll_json_result(stem, timeout=5, interval=2)
        if not result_json:
            continue
        masked_pages = fetch_masked_image(stem, timeout=5, interval=2)
        page_images = {}
        docs = result_json.get("result", {}).get("documents", [])
        for doc in docs:
            pn = doc.get("pageNo", 1)
            if pn not in page_images:
                img = fetch_page_image(stem, pn)
                if img:
                    page_images[pn] = img
        restored.append({
            "filename": filename,
            "stem": stem,
            "result_json": result_json,
            "masked_pages": masked_pages,
            "file_bytes": page_images.get(1, b""),
            "page_images": page_images,
        })
    if restored:
        st.session_state.results = restored


_restore_results_from_url()


def _divider():
    st.markdown('<hr style="margin:6px 0 20px 0;">', unsafe_allow_html=True)


# ── Upload section ────────────────────────────────────────────────────────────
st.markdown(
    '<div style="font-size:16px;font-weight:800;color:#0D1B2A;margin-bottom:10px;">Upload Documents</div>',
    unsafe_allow_html=True,
)
uploaded_files = st.file_uploader(
    label="Drop files here",
    type=["png", "jpg", "jpeg", "pdf"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

# File list preview
if uploaded_files:
    st.markdown(
        f'<div style="font-size:13px;color:#4A5568;margin:10px 0 4px 0;">'
        f'<b>{len(uploaded_files)}</b> file{"s" if len(uploaded_files) > 1 else ""} selected</div>',
        unsafe_allow_html=True,
    )
    for f in uploaded_files:
        size_kb = len(f.getvalue()) / 1024
        st.markdown(
            f'<div style="font-size:13px;color:#0D1B2A;padding:4px 0;">'
            f'📎 <b>{f.name}</b> &nbsp;<span style="color:#A0AEC0;">({size_kb:.1f} KB)</span></div>',
            unsafe_allow_html=True,
        )

# ── Masking options ───────────────────────────────────────────────────────
opt_col1, opt_col2, _ = st.columns([3, 3, 4])
with opt_col1:
    mask_digits = st.selectbox(
        "Aadhaar digits to mask",
        options=[8, 12],
        index=0,
        format_func=lambda v: f"{v} digits — {'last 4 visible' if v == 8 else 'fully masked'}",
    )

_divider()

# ── Process button ────────────────────────────────────────────────────────────
btn_col, clear_col, _ = st.columns([2, 1, 5])

with btn_col:
    process_clicked = st.button(
        "⚡ Process Documents",
        type="primary",
        disabled=not uploaded_files or st.session_state.processing,
        width="stretch",
    )

with clear_col:
    if st.button("🗑 Clear", width="stretch", disabled=st.session_state.processing):
        st.session_state.results = []
        st.query_params.clear()
        st.rerun()

# ── Processing pipeline ───────────────────────────────────────────────────────
STEPS = ["Upload", "OCR", "Classification", "Extraction", "Done"]

def _step_html(steps_done: int, is_aadhaar: bool = False) -> str:
    labels = ["Upload", "OCR", "Classify", "Extract"] + (["Masking"] if is_aadhaar else ["Done"])
    icons  = ["☁", "🔍", "🏷", "📋"] + (["🛡"] if is_aadhaar else ["✅"])
    parts = []
    for i, (label, icon) in enumerate(zip(labels, icons)):
        if i < steps_done:
            color, bg, fw = "#0A7C59", "#E6F4EF", "700"
        elif i == steps_done:
            color, bg, fw = "#1E5EBB", "#E8F0FB", "700"
        else:
            color, bg, fw = "#A0AEC0", "#F7F8FA", "500"
        parts.append(
            f'<span style="background:{bg};color:{color};font-size:12px;font-weight:{fw};'
            f'padding:4px 10px;border-radius:20px;border:1px solid {color}33;">'
            f'{icon} {label}</span>'
        )
    return '<div style="display:flex;gap:6px;flex-wrap:wrap;margin:6px 0;">' + "".join(parts) + "</div>"


if process_clicked and uploaded_files:
    st.session_state.processing = True
    st.session_state.results = []
    new_results = []

    try:
        for file in uploaded_files:
            file_bytes = file.getvalue()
            filename = file.name

            with st.container():
                st.markdown(
                    f'<div style="font-size:14px;font-weight:600;color:#0D1B2A;margin-bottom:6px;">📁 {filename}</div>',
                    unsafe_allow_html=True,
                )
                status_box = st.empty()

                # Step 0: Upload
                status_box.markdown(_step_html(0), unsafe_allow_html=True)
                try:
                    with st.spinner(f"Uploading {filename}…"):
                        stem = upload_file(file_bytes, filename, mask_digits=mask_digits)
                except Exception as e:
                    st.error(f"Upload failed: {e}")
                    continue

                # Step 1–3: OCR / classify / extract (all happen inside Lambda)
                status_box.markdown(_step_html(1), unsafe_allow_html=True)
                progress_bar = st.progress(0, text="Waiting for OCR pipeline…")

                result_json = None
                t_pipeline_start = time.time()
                timeout = 300
                interval = 5
                while time.time() - t_pipeline_start < timeout:
                    elapsed = time.time() - t_pipeline_start
                    pct = min(int((elapsed / timeout) * 85), 85)
                    progress_bar.progress(pct, text=f"Processing… ({int(elapsed)}s)")
                    try:
                        result_json = poll_json_result(stem, timeout=interval, interval=interval)
                    except Exception as e:
                        progress_bar.empty()
                        status_box.empty()
                        st.error(f"Error polling result for {filename}: {e}")
                        result_json = None
                        break
                    if result_json:
                        break
                    step = 1 if elapsed < 35 else 2 if elapsed < 60 else 3
                    status_box.markdown(_step_html(step), unsafe_allow_html=True)

                if not result_json:
                    progress_bar.empty()
                    status_box.empty()
                    st.error(f"⏱ Timeout: no result received for {filename} after 5 minutes.")
                    continue

                progress_bar.progress(90, text="Extraction complete")
                status_box.markdown(_step_html(3), unsafe_allow_html=True)

                # Step 4: Masking (Aadhaar only — fetched from Lambda via S3)
                masked_pages = None
                if is_aadhaar_result(result_json):
                    status_box.markdown(_step_html(3, is_aadhaar=True), unsafe_allow_html=True)
                    progress_bar.progress(92, text="Applying Aadhaar masking…")
                    masked_pages = fetch_masked_image(stem, timeout=120, interval=5)
                    if masked_pages:
                        progress_bar.progress(100, text="Masking complete ✔")
                    else:
                        progress_bar.progress(100, text="No masking regions detected")
                    status_box.markdown(_step_html(5, is_aadhaar=True), unsafe_allow_html=True)
                else:
                    progress_bar.progress(100, text="Done ✔")
                    status_box.markdown(_step_html(4), unsafe_allow_html=True)

                # Fetch per-page images for PDF preview
                page_images = {}
                docs = result_json.get("result", {}).get("documents", [])
                unique_pages = {doc.get("pageNo", 1) for doc in docs}
                for page_no in unique_pages:
                    img = fetch_page_image(stem, page_no)
                    if img:
                        page_images[page_no] = img

                new_results.append({
                    "filename": filename,
                    "stem": stem,
                    "result_json": result_json,
                    "masked_pages": masked_pages,
                    "file_bytes": file_bytes,
                    "page_images": page_images,
                })
    finally:
        st.session_state.results = new_results
        st.session_state.processing = False
        if new_results:
            encoded = ",".join(f"{r['stem']}:{r['filename']}" for r in new_results)
            st.query_params["r"] = encoded
        st.rerun()

# ── Results section ───────────────────────────────────────────────────────────
if st.session_state.results:
    results = st.session_state.results

    # Summary bar
    total = len(results)
    aadhaar_count = sum(1 for r in results if is_aadhaar_result(r["result_json"]))
    st.markdown(
        f"""
<div style="background:#FFFFFF;border-radius:10px;border:1px solid #E2E8F0;
     padding:14px 24px;margin-bottom:20px;display:flex;gap:32px;align-items:center;">
  <div>
    <div style="font-size:22px;font-weight:700;color:#0D1B2A;">{total}</div>
    <div style="font-size:12px;color:#7A8499;">Documents Processed</div>
  </div>
  <div>
    <div style="font-size:22px;font-weight:700;color:#1E5EBB;">{aadhaar_count}</div>
    <div style="font-size:12px;color:#7A8499;">Aadhaar (Masked)</div>
  </div>
  <div>
    <div style="font-size:22px;font-weight:700;color:#0A7C59;">{total - aadhaar_count}</div>
    <div style="font-size:12px;color:#7A8499;">Other Documents</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Tabs: one per file (if batch) or single
    if len(results) == 1:
        r = results[0]
        render_full_result(r["result_json"], r["file_bytes"], r.get("masked_pages"), r["filename"],
                           r.get("page_images", {}), result_index=0)
    else:
        # Deduplicate tab labels for repeated filenames
        name_counts: dict[str, int] = {}
        tab_labels = []
        for r in results:
            name = r["filename"]
            name_counts[name] = name_counts.get(name, 0) + 1
            if name_counts[name] > 1:
                tab_labels.append(f"{name} ({name_counts[name]})")
            else:
                tab_labels.append(name)

        tabs = st.tabs(tab_labels)
        for idx, (tab, r) in enumerate(zip(tabs, results)):
            with tab:
                render_full_result(r["result_json"], r["file_bytes"], r.get("masked_pages"), r["filename"],
                                   r.get("page_images", {}), result_index=idx)

    # Download all as ZIP
    _divider()
    if len(results) > 1:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for r in results:
                stem = Path(r["filename"]).stem
                zf.writestr(
                    f"{stem}_result.json",
                    json.dumps(r["result_json"], indent=2, ensure_ascii=False),
                )
        st.download_button(
            label="⬇ Download All Results (ZIP)",
            data=zip_buf.getvalue(),
            file_name="document_results.zip",
            mime="application/zip",
        )

# ── Empty state ───────────────────────────────────────────────────────────────
if not uploaded_files and not st.session_state.results:
    st.markdown(
        """
<div style="text-align:center;padding:60px 20px;color:#A0AEC0;">
  <div style="font-size:48px;margin-bottom:12px;">📂</div>
  <div style="font-size:16px;font-weight:600;color:#4A5568;">Upload KYC documents to get started</div>
  <div style="font-size:13px;margin-top:6px;">
    Automated OCR, document classification &amp; Aadhaar masking — single or batch
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div style="text-align:center;padding:24px 0 12px 0;margin-top:40px;border-top:1px solid #E2E8F0;">
  <span style="color:#A0AEC0;font-size:12px;">&copy; 2026 Aivar Innovations. All rights reserved.</span>
</div>
""",
    unsafe_allow_html=True,
)
