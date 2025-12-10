
"""
Full Integrated Streamlit App:
- Structural defect detection (cracks, delamination, corrosion, fractures)
- Input options: upload image, real-time camera, live video feed
- Edge detection + Thermal imaging
- YOLO detection with red boxes and labels
- Live parameter tuning
- Modern, easy-to-use dashboard
"""

import streamlit as st
import numpy as np
import cv2
import os
from datetime import datetime
from ultralytics import YOLO
from PIL import Image
import time
import math

# ----------------------
# --- Configuration ---
# ----------------------
SAVE_DIR = "saved_defect_images"
os.makedirs(SAVE_DIR, exist_ok=True)
st.set_page_config(layout="wide", page_title="Aircraft Wing Defect Detection")

# Initialize session state
if "webcam_running" not in st.session_state:
    st.session_state.webcam_running = False
if "cap" not in st.session_state:
    st.session_state.cap = None
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None

# ----------------------
# --- Sidebar UI ---
# ----------------------
st.sidebar.title("Input & Controls")
input_mode = st.sidebar.radio(
    "Input mode", ["Upload Image", "OpenCV Webcam", "Live Video Feed"])
uploaded_file = None
if input_mode == "Upload Image":
    uploaded_file = st.sidebar.file_uploader(
        "Upload image (.png/.jpg/.jpeg)", type=["png", "jpg", "jpeg"])

# Webcam controls
st.sidebar.markdown("### Webcam Controls")
colw1, colw2 = st.sidebar.columns(2)
with colw1:
    start_cam = st.button("Start Camera")
with colw2:
    stop_cam = st.button("Stop Camera")

# Save images checkbox
save_images_checkbox = st.sidebar.checkbox("Save Images", value=True)

# Thermal and detection controls
st.sidebar.header("Detection & Thermal Imaging")
palette_choice = st.sidebar.selectbox(
    "Thermal palette", ["Red-hot", "Green (Viridis)", "White-hot"])
det_method = st.sidebar.selectbox(
    "Detection method", ["YOLO", "Edges", "Contours"])

# Image preprocessing
st.sidebar.header("Preprocessing")
normalize_min = st.sidebar.slider("Normalize Min Percentile", 0, 100, 1)
normalize_max = st.sidebar.slider("Normalize Max Percentile", 1, 100, 99)
apply_blur = st.sidebar.checkbox("Apply Gaussian Blur", value=True)
blur_ksize = st.sidebar.slider("Blur Kernel (odd)", 1, 31, 3)
if blur_ksize % 2 == 0:
    blur_ksize += 1

# Canny / Hough
st.sidebar.header("Edge & X Detection")
canny_min = st.sidebar.slider("Canny Threshold1", 0, 500, 50)
canny_max = st.sidebar.slider("Canny Threshold2", 0, 500, 150)

# YOLO Model
st.sidebar.header("YOLO Model")
model_choice = st.sidebar.selectbox(
    "YOLO Model", ["yolov8n.pt"])  # placeholder
st.sidebar.markdown(
    "Use a pre-trained model; replace with your trained aircraft defect model for best accuracy.")

# ----------------------
# --- YOLO Load ---
# ----------------------


@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path):
    return YOLO(model_path)


model = load_yolo_model(model_choice)

# ----------------------
# --- Utility Functions ---
# ----------------------


def to_uint8(arr):
    a = arr.astype(np.float32)
    a_min, a_max = a.min(), a.max()
    if a_max - a_min == 0:
        return np.zeros_like(a, dtype=np.uint8)
    return ((arr - a_min)/(a_max - a_min)*255).astype(np.uint8)


def normalize_percentiles(arr, pmin, pmax):
    low = np.percentile(arr, pmin)
    high = np.percentile(arr, pmax)
    arr = np.clip(arr, low, high)
    return ((arr - low)/(high-low)*255).astype(np.uint8)


def apply_thermal_palette(gray, palette):
    if palette == "Red-hot":
        return cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
    elif palette == "Green (Viridis)":
        try:
            return cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
        except:
            return cv2.applyColorMap(gray, cv2.COLORMAP_OCEAN)
    else:
        lut = np.array([min(255, int(((i/255.0)**0.5)*255))
                       for i in range(256)], dtype=np.uint8)
        mapped = cv2.LUT(gray, lut)
        base = cv2.applyColorMap(mapped, cv2.COLORMAP_BONE)
        return cv2.convertScaleAbs(base, alpha=1.3, beta=10)


def detect_edges(gray):
    return cv2.Canny(gray, canny_min, canny_max)


def detect_contours(gray):
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(
        th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def detect_yolo(img):
    results = model(img)[0]
    overlay = img.copy()
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        label = f"{model.names[int(cls)]} {conf:.2f}"
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(overlay, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return overlay

# ----------------------
# --- Webcam Functions ---
# ----------------------


def start_webcam():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        st.warning("Cannot open camera.")
        return None
    st.session_state.cap = cap
    st.session_state.webcam_running = True
    return cap


def stop_webcam():
    if st.session_state.cap is not None:
        st.session_state.cap.release()
    st.session_state.cap = None
    st.session_state.webcam_running = False


if start_cam:
    if not st.session_state.webcam_running:
        start_webcam()
if stop_cam:
    stop_webcam()

# ----------------------
# --- Main Layout ---
# ----------------------
st.title("Aircraft Wing Structural Defect Detection Dashboard")
st.markdown("Upload images, take pictures, or analyze live video feed. YOLO detection, edge detection, thermal mapping, and live parameter tuning.")

left_col, mid_col, right_col = st.columns([1, 1, 1])
orig_ph = left_col.empty()
thermal_ph = mid_col.empty()
detect_ph = right_col.empty()

# Load source image


def load_image(uploaded, last_frame):
    img = None
    name = None
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        name = uploaded.name
    elif last_frame is not None:
        img = last_frame.copy()
        name = f"webcam_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
    return img, name


# Capture webcam frame
if st.session_state.webcam_running and st.session_state.cap:
    ret, frame = st.session_state.cap.read()
    if ret:
        st.session_state.last_frame = frame.copy()

src_img, src_name = load_image(uploaded_file, st.session_state.last_frame)
if src_img is None:
    st.info("Upload image or start webcam.")
    st.stop()

# Save image if requested
if save_images_checkbox and src_name:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    save_path = os.path.join(SAVE_DIR, f"{timestamp}_{src_name}")
    cv2.imwrite(save_path, src_img)
    st.sidebar.success(f"Saved: {save_path}")

# Preprocessing
gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
gray = normalize_percentiles(gray, normalize_min, normalize_max)
if apply_blur and blur_ksize > 1:
    gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

# Thermal
thermal_bgr = apply_thermal_palette(gray, palette_choice)
thermal_rgb = cv2.cvtColor(thermal_bgr, cv2.COLOR_BGR2RGB)

# Detection
if det_method == "Edges":
    edges = detect_edges(gray)
    detect_overlay = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
elif det_method == "Contours":
    contours = detect_contours(gray)
    detect_overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(detect_overlay, contours, -1, (0, 255, 0), 2)
else:
    detect_overlay = detect_yolo(src_img)

# Display
orig_ph.image(gray, caption="Grayscale",
              channels="GRAY", use_column_width=True)
thermal_ph.image(
    thermal_rgb, caption=f"Thermal ({palette_choice})", use_column_width=True)
detect_ph.image(cv2.cvtColor(detect_overlay, cv2.COLOR_BGR2RGB),
                caption=f"{det_method}", use_column_width=True)

# Download buttons
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    _, buf = cv2.imencode(".png", thermal_rgb)
    st.download_button("Download Thermal PNG", buf.tobytes(),
                       file_name="thermal.png", mime="image/png")
with col2:
    _, buf = cv2.imencode(".png", cv2.cvtColor(
        detect_overlay, cv2.COLOR_BGR2RGB))
    st.download_button("Download Detection PNG", buf.tobytes(),
                       file_name="detection.png", mime="image/png")

# Saved images gallery
st.markdown("### Saved Images")
saved = sorted(os.listdir(SAVE_DIR), reverse=True)
if saved:
    cols = st.columns(6)
    for idx, f in enumerate(saved[:24]):
        with cols[idx % 6]:
            st.image(os.path.join(SAVE_DIR, f),
                     use_column_width=True, caption=f)

# Automatic rerun for webcam feed
if st.session_state.webcam_running:
    time.sleep(0.03)
    st.experimental_rerun()
