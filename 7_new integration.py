
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageEnhance
import tempfile
import time

# ---------------------------
# FUTURISTIC UI CONFIG
# ---------------------------
st.set_page_config(
    page_title="Structure Automator",
    page_icon="🛩️",
    layout="wide",
)

# Custom futuristic CSS
st.markdown("""
<style>
body {
    background-color: #0e0f1a;
    color: #e4e7ef;
    font-family: 'Segoe UI', sans-serif;
}
.sidebar .sidebar-content {
    background-color: #11121d !important;
}
.block-container {
    padding-top: 1rem;
}
.neon-title {
    font-size: 36px !important;
    font-weight: 700;
    text-shadow: 0px 0px 12px #00f0ff;
    color: #00f0ff;
}
.card {
    padding:16px;
    background: #141524;
    border-radius: 12px;
    border: 1px solid #24273a;
    box-shadow: 0 0 12px rgba(0,255,255,0.06);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# LOAD MODEL
# ---------------------------


@st.cache_resource
def load_model():
    try:
        model = YOLO("yolov8n-seg.pt")  # segmentation-capable YOLO
    except Exception:
        model = YOLO("yolov8n.pt")  # fallback
    return model


model = load_model()

# ---------------------------
# IMAGE ENHANCEMENT
# ---------------------------


def apply_enhancements(img, brightness, contrast, denoise):
    pil_img = Image.fromarray(img)
    pil_img = ImageEnhance.Brightness(pil_img).enhance(brightness)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(contrast)
    img = np.array(pil_img)

    if denoise > 0:
        img = cv2.fastNlMeansDenoisingColored(
            img, None, denoise, denoise, 7, 21)

    return img

# ---------------------------
# DEFECT CLASSIFICATION HEURISTIC
# ---------------------------


def classify_defect(box_w, box_h, sharp_edges):
    area = box_w * box_h

    if area < 400:
        return "Candidate (Uncertain—small region)"

    if sharp_edges > 20:
        return "Crack"

    if 400 < area < 3000:
        return "Corrosion"

    if area > 3000:
        return "Delamination"

    return "Candidate"

# ---------------------------
# PROCESS YOLO RESULTS
# ---------------------------


def analyze_image(image, conf):
    results = model.predict(image, conf=conf)
    annotated = image.copy()
    suggestions = []

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf_score = float(box.conf[0])
            cls = int(box.cls[0])

            crop = image[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 40, 120)
            sharp_edges = np.sum(edges > 0)

            defect_type = classify_defect(x2-x1, y2-y1, sharp_edges)

            color = (0, 255, 255) if defect_type != "Candidate" else (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            cv2.putText(annotated, f"{defect_type} {conf_score:.2f}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            suggestions.append({
                "type": defect_type,
                "confidence": conf_score,
                "area": (x2-x1)*(y2-y1)
            })

    return annotated, suggestions


# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.title("🛩️ Structure Automator")
input_type = st.sidebar.radio("Input type:", ["Upload Image", "Use Camera"])

aircraft_part = st.sidebar.selectbox(
    "Select Aircraft Section:",
    ["Wing", "Fuselage", "Empennage", "Landing Gear", "Engine Inlet"]
)

# Enhancements
st.sidebar.markdown("### Image Enhancement")
brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.0)
contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0)
denoise = st.sidebar.slider("Denoise", 0, 20, 0)

conf = st.sidebar.slider("Detection Confidence", 0.2, 0.9, 0.45)

# ---------------------------
# MAIN UI
# ---------------------------
st.markdown("<h1 class='neon-title'>STRUCTURE AUTOMATOR</h1>",
            unsafe_allow_html=True)
st.write("Futuristic AI-powered airframe defect detection using YOLO + thermal/visual imaging.")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### 📤 Input Feed")

    file = None
    if input_type == "Upload Image":
        file = st.file_uploader("Upload an Aircraft Image", type=[
                                "jpg", "png", "jpeg"])
    else:
        file = st.camera_input("Take image")

    if file:
        img = Image.open(file)
        img = np.array(img)

        enhanced = apply_enhancements(img, brightness, contrast, denoise)

        annotated, suggestions = analyze_image(enhanced, conf)

        st.markdown("### 🔍 Enhanced & Analyzed Output")
        st.image(annotated, caption=f"Detected Defects - {aircraft_part}")

with col2:
    st.markdown("### 📁 Analysis Summary")
    if file:
        for s in suggestions:
            st.markdown(f"""
            <div class='card'>
            <b>Defect Type:</b> {s['type']}<br>
            <b>Confidence:</b> {s['confidence']:.2f}<br>
            <b>Area:</b> {s['area']} px²<br>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("Upload an image to see detection results.")

# ---------------------------
# END
# ---------------------------
