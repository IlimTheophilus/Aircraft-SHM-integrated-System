import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import io
from skimage import filters, feature, morphology, measure, color

st.set_page_config(layout="wide")

# ── DESIGN SYSTEM ────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --bg: #0B1220;
    --surface: #111827;
    --glass: rgba(255,255,255,0.04);
    --border: rgba(255,255,255,0.08);
    --accent: #06B6D4;
    --danger: #EF4444;
}

.stApp { background: linear-gradient(135deg, #0B1220, #0F172A); }

.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem;
}

.metric {
    background: var(--glass);
    border: 1px solid var(--border);
    padding: 1rem;
    border-radius: 12px;
    text-align:center;
}

.metric-value {
    font-size:1.4rem;
    font-weight:700;
    color: var(--accent);
}

#MainMenu, footer, header {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── FUNCTIONS ────────────────────────────────────────────────────────────
def detect(rgb):
    gray = color.rgb2gray(rgb)
    edges = feature.canny(gray)
    closed = morphology.closing(edges, morphology.square(3))
    labels = measure.label(closed)

    boxes = []
    for r in measure.regionprops(labels):
        if r.area > 50:
            y1, x1, y2, x2 = r.bbox
            boxes.append((x1,y1,x2,y2))
    return boxes

def draw_boxes(img, boxes):
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    for b in boxes:
        draw.rectangle(b, outline=(239,68,68), width=2)
    return pil

# ── UI ───────────────────────────────────────────────────────────────────
st.markdown("## 🔍 Structural Detection Dashboard")

col1, col2 = st.columns([1,2])

with col1:
    img_file = st.file_uploader("Upload Image")

with col2:
    if img_file:
        image = Image.open(img_file).convert("RGB")
        arr = np.array(image)

        boxes = detect(arr)
        annotated = draw_boxes(arr, boxes)

        m1, m2 = st.columns(2)

        with m1:
            st.image(image, caption="Original")

        with m2:
            st.image(annotated, caption="Detected")

        st.markdown("### Metrics")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown(f"""
            <div class="metric">
                <div class="metric-value">{len(boxes)}</div>
                <div>Defects</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div class="metric">
                <div class="metric-value">Active</div>
                <div>Status</div>
            </div>
            """, unsafe_allow_html=True)
