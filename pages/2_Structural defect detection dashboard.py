import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
from skimage import exposure, filters, feature, morphology, measure, color

st.set_page_config(page_title="ASHMIS SDD Dashboard", layout="wide", page_icon="🔍")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
.stApp { background: #020c1b; color: #e2e8f0; }
.block-container { padding: 1.5rem 2rem !important; }
section[data-testid="stSidebar"] { background: #0a1628 !important; border-right: 1px solid #0ea5e920; }
.sdd-header {
    background: linear-gradient(135deg, #0a1628, #0f2744);
    border: 1px solid #0ea5e920; border-radius: 12px;
    padding: 1.8rem 2rem; margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}
.sdd-header::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg, transparent, #0ea5e9, #06b6d4, transparent);
}
.sdd-title { font-family: Orbitron, monospace; font-size: 1.4rem; font-weight: 700; color: #fff; }
.sdd-title span { color: #0ea5e9; }
.sdd-subtitle { font-family: Inter, sans-serif; font-size: 0.8rem; color: #64748b; letter-spacing: 1px; margin-top: 0.3rem; }
.sdd-header-top { display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem; }
.sdd-badges { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 0.5rem; }
.sdd-badge {
    background: rgba(14,165,233,0.1); border: 1px solid rgba(14,165,233,0.25);
    color: #38bdf8; padding: 0.25rem 0.7rem; border-radius: 4px;
    font-size: 0.68rem; font-family: Inter, sans-serif;
    letter-spacing: 1px; text-transform: uppercase; font-weight: 600;
}
.panel { background: #0a1628; border: 1px solid #0ea5e915; border-radius: 12px; padding: 1.4rem; }
.panel-header {
    font-family: Orbitron, monospace; font-size: 0.65rem; color: #38bdf8;
    letter-spacing: 3px; text-transform: uppercase; margin-bottom: 1.2rem;
}
.param-group { background: #0f2744; border: 1px solid #0ea5e910; border-radius: 8px; padding: 1rem; margin-bottom: 0.8rem; }
.param-group-title { font-family: Orbitron, monospace; font-size: 0.6rem; color: #38bdf8; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 0.8rem; }
.metric-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.8rem; margin: 1rem 0; }
.metric-card { background: #0f2744; border: 1px solid #0ea5e915; border-radius: 8px; padding: 1rem; text-align: center; }
.metric-val { font-family: Orbitron, monospace; font-size: 1.6rem; font-weight: 700; color: #0ea5e9; }
.metric-lbl { font-family: Inter, sans-serif; font-size: 0.65rem; color: #64748b; letter-spacing: 1.5px; text-transform: uppercase; margin-top: 0.3rem; }
.detect-status { padding: 0.6rem 1.2rem; border-radius: 8px; font-family: Orbitron, monospace; font-size: 0.75rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; text-align: center; margin: 0.8rem 0; }
.detect-found { background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.4); color: #f87171; }
.detect-clear { background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.4); color: #34d399; }
.detect-table { width: 100%; border-collapse: collapse; font-family: JetBrains Mono, monospace; font-size: 0.78rem; margin-top: 0.8rem; }
.detect-table th { background: #0f2744; color: #38bdf8; padding: 0.5rem 0.8rem; text-align: left; font-size: 0.65rem; letter-spacing: 1px; text-transform: uppercase; border-bottom: 1px solid #0ea5e920; }
.detect-table td { padding: 0.5rem 0.8rem; color: #94a3b8; border-bottom: 1px solid #0ea5e908; }
.col-crack { color: #f87171; font-weight: 700; }
.col-corrosion { color: #fb923c; font-weight: 700; }
.col-delamination { color: #fbbf24; font-weight: 700; }
.col-candidate { color: #94a3b8; }
.img-label { font-family: Orbitron, monospace; font-size: 0.6rem; color: #38bdf8; letter-spacing: 2px; margin-bottom: 0.5rem; }
.info-msg { background: rgba(14,165,233,0.06); border: 1px solid rgba(14,165,233,0.15); border-left: 3px solid #0ea5e9; border-radius: 6px; padding: 0.8rem 1rem; font-family: Inter, sans-serif; font-size: 0.82rem; color: #64748b; margin: 0.5rem 0; }
.await-box { text-align: center; padding: 4rem 2rem; }
.await-icon { font-size: 3rem; margin-bottom: 1rem; }
.await-title { font-family: Orbitron, monospace; font-size: 0.75rem; color: #38bdf8; letter-spacing: 3px; text-transform: uppercase; }
.await-desc { font-family: Inter, sans-serif; font-size: 0.85rem; color: #475569; margin-top: 0.8rem; }
.stButton > button { background: linear-gradient(135deg, #0369a1, #0ea5e9) !important; color: white !important; border: none !important; border-radius: 8px !important; font-family: Inter, sans-serif !important; font-weight: 600 !important; }
.stDownloadButton > button { background: rgba(16,185,129,0.15) !important; color: #34d399 !important; border: 1px solid rgba(16,185,129,0.3) !important; border-radius: 8px !important; }
label { color: #94a3b8 !important; font-family: Inter, sans-serif !important; font-size: 0.8rem !important; }
.page-footer { text-align: center; padding: 1.5rem; margin-top: 1rem; font-family: Inter, sans-serif; font-size: 0.72rem; color: #334155; border-top: 1px solid #0ea5e910; letter-spacing: 1px; }

/* Hide streamlit chrome but KEEP sidebar toggle button visible */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


def pil_to_array(img):
    return np.array(img.convert("RGB"))

def array_to_pil(arr):
    return Image.fromarray(arr.astype(np.uint8))

def draw_annotations(rgb_arr, candidates, show_labels=True):
    pil = array_to_pil(rgb_arr)
    draw = ImageDraw.Draw(pil)
    color_map = {
        "crack": (239, 68, 68),
        "corrosion": (251, 146, 60),
        "delamination": (251, 191, 36),
        "candidate": (148, 163, 184)
    }
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for c in candidates:
        x0, y0, x1, y1 = c["box"]
        col = color_map.get(c.get("label", "candidate"), (148, 163, 184))
        draw.rectangle([x0, y0, x1, y1], outline=col, width=2)
        if show_labels:
            lbl = c.get("label", "candidate")
            if c.get("score") is not None:
                lbl = f"{lbl} {c['score']:.2f}"
            if font:
                bbox = draw.textbbox((0, 0), lbl, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            else:
                tw = len(lbl) * 6
                th = 10
            tx = x0
            ty = max(0, y0 - th - 3)
            draw.rectangle([tx, ty, tx + tw + 4, ty + th + 2], fill=(0, 0, 0))
            draw.text((tx + 2, ty + 1), lbl, fill=col, font=font)
    return np.array(pil)

def detect_candidates(rgb_arr, params):
    gray = color.rgb2gray(rgb_arr)
    try:
        clahe = exposure.equalize_adapthist(
            gray,
            clip_limit=max(0.01, params["clahe_clip"] / 8.0),
            kernel_size=(params["clahe_tile"], params["clahe_tile"])
        )
    except TypeError:
        clahe = exposure.equalize_adapthist(
            gray, clip_limit=max(0.01, params["clahe_clip"] / 8.0)
        )
    blur_k = params["blur_k"]
    blurred = filters.gaussian(clahe, sigma=max(0.3, blur_k / 2.0)) if blur_k > 1 else clahe
    low_t = np.clip(params["canny_low"] / 255.0, 0.0, 1.0)
    high_t = np.clip(params["canny_high"] / 255.0, 0.0, 1.0)
    try:
        edges = feature.canny(blurred, low_threshold=low_t, high_threshold=high_t)
    except TypeError:
        edges = feature.canny(blurred, sigma=1.0)
    morph_k = max(1, int(params["morph_k"]))
    selem = morphology.square(morph_k)
    closed = edges.copy()
    for _ in range(max(1, params["morph_iter"])):
        closed = morphology.closing(closed, selem)
    label_img = measure.label(closed)
    regions = measure.regionprops(label_img)
    h, w = gray.shape
    candidates = []
    for region in regions:
        area = region.area
        if area < params["min_area"] or area > params["max_area"]:
            continue
        minr, minc, maxr, maxc = region.bbox
        pad = int(max(2, 0.02 * max(w, h)))
        x0 = max(0, minc - pad)
        y0 = max(0, minr - pad)
        x1 = min(w - 1, maxc + pad)
        y1 = min(h - 1, maxr + pad)
        ww = max(1, x1 - x0)
        hh = max(1, y1 - y0)
        aspect = ww / (hh + 1e-8)
        mean_int = int(np.mean(clahe[y0:y1, x0:x1]) * 255) if (y1 > y0 and x1 > x0) else 255
        if aspect > params["aspect_crack_thresh"]:
            label, score = "crack", min(1.0, area / (params["max_area"] + 1e-6))
        elif mean_int < params["dark_thresh"] and area > params["delam_area_thresh"]:
            label, score = "corrosion", min(1.0, area / (params["max_area"] + 1e-6))
        elif area > params["delam_area_thresh"]:
            label, score = "delamination", min(1.0, area / (params["max_area"] + 1e-6))
        else:
            label, score = "candidate", None
        candidates.append({
            "box": (x0, y0, x1, y1),
            "area": area,
            "label": label,
            "score": score
        })
    edge_map = (closed.astype(np.uint8) * 255)
    return edge_map, candidates


if "stored_image_bytes" not in st.session_state:
    st.session_state.stored_image_bytes = None

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sdd-header">
    <div class="sdd-header-top">
        <div>
            <div class="sdd-title">ASHMIS <span>SDD</span> Dashboard</div>
            <div class="sdd-subtitle">STRUCTURAL DEFECT DETECTION &mdash; SNAPSHOT + LIVE PARAMETER TUNING</div>
        </div>
        <div class="sdd-badges">
            <span class="sdd-badge">VARIANT A</span>
            <span class="sdd-badge">AI-Assisted</span>
            <span class="sdd-badge">Real-Time</span>
            <span class="sdd-badge">NCAA Compliant</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 2], gap="medium")

with left_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-header">IMAGE ACQUISITION</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-msg">Capture with webcam or upload an image file to begin live analysis.</div>', unsafe_allow_html=True)

    cam = st.camera_input("Live Webcam Capture")
    uploaded = st.file_uploader("Upload Image File", type=["jpg", "jpeg", "png", "bmp"])

    if cam is not None:
        st.session_state.stored_image_bytes = cam.getvalue()
    elif uploaded is not None:
        st.session_state.stored_image_bytes = uploaded.getvalue()

    if st.session_state.stored_image_bytes:
        try:
            preview = Image.open(io.BytesIO(st.session_state.stored_image_bytes))
            st.image(preview, caption="Stored Source Image", use_container_width=True)
        except Exception as e:
            st.error(f"Image load error: {e}")
    else:
        st.markdown('<div class="info-msg">No image loaded yet.</div>', unsafe_allow_html=True)

    if st.button("Clear Image", use_container_width=True):
        st.session_state.stored_image_bytes = None
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


with right_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-header">DETECTION PARAMETERS &mdash; LIVE TUNING</div>', unsafe_allow_html=True)

    p_col1, p_col2, p_col3 = st.columns(3)

    with p_col1:
        st.markdown('<div class="param-group"><div class="param-group-title">Image Enhancement</div>', unsafe_allow_html=True)
        clahe_clip = st.slider("CLAHE Clip Limit", 1.0, 8.0, 2.0, 0.1)
        clahe_tile = st.selectbox("CLAHE Tile Grid", [4, 8, 16], index=1)
        blur_k = st.selectbox("Gaussian Blur Kernel", [1, 3, 5, 7, 9], index=1)
        st.markdown('</div>', unsafe_allow_html=True)

    with p_col2:
        st.markdown('<div class="param-group"><div class="param-group-title">Edge Detection</div>', unsafe_allow_html=True)
        canny_low = st.slider("Canny Low Threshold", 1, 255, 50)
        canny_high = st.slider("Canny High Threshold", 1, 500, 150)
        morph_k = st.selectbox("Morph Kernel Size", [3, 5, 7, 9], index=0)
        morph_iter = st.slider("Morph Iterations", 0, 5, 1)
        st.markdown('</div>', unsafe_allow_html=True)

    with p_col3:
        st.markdown('<div class="param-group"><div class="param-group-title">Defect Classifier</div>', unsafe_allow_html=True)
        min_area = st.number_input("Min Contour Area (px)", 1, 100000, 50, 1)
        max_area = st.number_input("Max Contour Area (px)", 100, 10000000, 8000, 100)
        aspect_crack_thresh = st.slider("Crack Aspect Ratio", 1.5, 50.0, 6.0, 0.5)
        dark_thresh = st.slider("Corrosion Dark Threshold", 0, 255, 60)
        delam_area_thresh = st.number_input("Delamination Area Threshold (px)", 10, 100000, 1500, 10)
        st.markdown('</div>', unsafe_allow_html=True)

    d_col1, d_col2, d_col3 = st.columns(3)
    with d_col1:
        show_labels = st.checkbox("Show Labels", value=True)
    with d_col2:
        show_edge_map = st.checkbox("Show Edge Map", value=False)
    with d_col3:
        download_name = st.text_input("Download Filename", "ASHMIS_annotated.jpg")

    st.markdown("<hr style='border:none;border-top:1px solid #0ea5e910;margin:1rem 0;'>", unsafe_allow_html=True)

    params = {
        "clahe_clip": float(clahe_clip),
        "clahe_tile": int(clahe_tile),
        "blur_k": int(blur_k),
        "canny_low": int(canny_low),
        "canny_high": int(canny_high),
        "morph_k": int(morph_k),
        "morph_iter": int(morph_iter),
        "min_area": int(min_area),
        "max_area": int(max_area),
        "aspect_crack_thresh": float(aspect_crack_thresh),
        "dark_thresh": int(dark_thresh),
        "delam_area_thresh": int(delam_area_thresh)
    }

    if st.session_state.stored_image_bytes:
        try:
            pil_img = Image.open(io.BytesIO(st.session_state.stored_image_bytes)).convert("RGB")
            rgb_arr = pil_to_array(pil_img)
            max_side = 1600
            h, w = rgb_arr.shape[:2]
            if max(h, w) > max_side:
                scale = max_side / float(max(h, w))
                pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                rgb_arr = pil_to_array(pil_img)

            edge_map, candidates = detect_candidates(rgb_arr, params)
            annotated_arr = draw_annotations(rgb_arr, candidates, show_labels)

            n_cracks = sum(1 for c in candidates if c["label"] == "crack")
            n_corr   = sum(1 for c in candidates if c["label"] == "corrosion")
            n_delam  = sum(1 for c in candidates if c["label"] == "delamination")
            n_total  = len(candidates)

            st.markdown(f"""
<div class="metric-row">
    <div class="metric-card">
        <div class="metric-val" style="color:#f87171;">{n_cracks}</div>
        <div class="metric-lbl">Cracks</div>
    </div>
    <div class="metric-card">
        <div class="metric-val" style="color:#fb923c;">{n_corr}</div>
        <div class="metric-lbl">Corrosion</div>
    </div>
    <div class="metric-card">
        <div class="metric-val" style="color:#fbbf24;">{n_delam}</div>
        <div class="metric-lbl">Delamination</div>
    </div>
    <div class="metric-card">
        <div class="metric-val">{n_total}</div>
        <div class="metric-lbl">Total Candidates</div>
    </div>
</div>
""", unsafe_allow_html=True)

            if n_total > 0:
                st.markdown(
                    f'<div class="detect-status detect-found">'
                    f'WARNING: STRUCTURAL ANOMALIES DETECTED &mdash; {n_total} CANDIDATE(S) FOUND'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="detect-status detect-clear">'
                    'SCAN COMPLETE &mdash; NO ANOMALIES DETECTED'
                    '</div>',
                    unsafe_allow_html=True
                )

            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.markdown('<div class="img-label">ORIGINAL IMAGE</div>', unsafe_allow_html=True)
                st.image(pil_img, use_container_width=True)
            with img_col2:
                st.markdown('<div class="img-label">ANNOTATED &mdash; LIVE DETECTION</div>', unsafe_allow_html=True)
                st.image(array_to_pil(annotated_arr), use_container_width=True)

            if show_edge_map:
                st.markdown('<div class="img-label" style="margin-top:1rem;">EDGE MAP &mdash; CLAHE / CANNY / MORPH</div>', unsafe_allow_html=True)
                st.image(Image.fromarray(edge_map), use_container_width=True)

            if candidates:
                rows_html = ""
                for i, c in enumerate(candidates[:12]):
                    lbl = c.get("label", "candidate")
                    score = f"{c['score']:.3f}" if c.get("score") is not None else "--"
                    rows_html += (
                        f"<tr>"
                        f"<td>{i + 1}</td>"
                        f"<td class='col-{lbl}'>{lbl.upper()}</td>"
                        f"<td>{int(c.get('area', 0))} px2</td>"
                        f"<td>{score}</td>"
                        f"<td style='font-size:0.7rem;'>{c['box']}</td>"
                        f"</tr>"
                    )
                st.markdown(
                    "<table class='detect-table'>"
                    "<thead><tr>"
                    "<th>#</th><th>Defect Type</th><th>Area</th><th>Score</th><th>Bounding Box</th>"
                    "</tr></thead>"
                    f"<tbody>{rows_html}</tbody>"
                    "</table>",
                    unsafe_allow_html=True
                )

            st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)
            buf = io.BytesIO()
            array_to_pil(annotated_arr).save(buf, format="JPEG")
            buf.seek(0)
            st.download_button(
                label="Download Annotated Image",
                data=buf,
                file_name=download_name,
                mime="image/jpeg",
                use_container_width=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Processing error: {e}")

    else:
        st.markdown("""
<div class="await-box">
    <div class="await-icon">📡</div>
    <div class="await-title">AWAITING IMAGE INPUT</div>
    <div class="await-desc">Capture with webcam or upload an image to begin structural analysis.</div>
</div>
""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="page-footer">
    ASHMIS SDD MODULE &mdash; AFIT KADUNA &nbsp;|&nbsp;
    NCAA &bull; EASA Part-145 &bull; ICAO &nbsp;|&nbsp;
    Ilim .A. Theophilus &copy; 2026 &nbsp;|&nbsp;
    Variant A &mdash; Heuristic Engine (ML Integration In Progress)
</div>
""", unsafe_allow_html=True)
