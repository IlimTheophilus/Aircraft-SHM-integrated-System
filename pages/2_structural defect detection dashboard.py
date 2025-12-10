# ashmi_interactive_variantA.py
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io

st.set_page_config(page_title="ASHMIS Interactive - Variant A", layout="wide")
st.title("ASHMIS Interactive — Snapshot + Live Parameter Tuning (Variant A)")

st.text("This Dashboard allows you to upload or capture live images, automatically analyse them using AI driven thermal/visual"
        " Inspection Models and adjust key detection parameters in real-time for more accurate results.")

st.subheader("""
Take a Picture in Realtime  or Upload an Image for Structural Health Analysis.
     """)

st.text("""The original image will be stored.
Adjust the image-processing sliders and the processed/annotated image updates **automatically**.
No need to press a 'Run' button each time.
""")

# --- Helpers ---


def pil_to_bgr(pil_img: Image.Image):
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr_img: np.ndarray):
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def annotate_image(bgr, candidates, show_labels=True):
    out = bgr.copy()
    for c in candidates:
        x0, y0, x1, y1 = c["box"]
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 2)  # red box
        if show_labels:
            lbl = c.get("label", "candidate")
            if c.get("score") is not None:
                lbl = f"{lbl} {c['score']:.2f}"
            cv2.putText(out, lbl, (x0, max(12, y0-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def detect_candidates(bgr, params):
    # Preprocess
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=params["clahe_clip"], tileGridSize=(
        params["clahe_tile"], params["clahe_tile"]))
    gray = clahe.apply(gray)
    blur_k = params["blur_k"]
    if blur_k > 1:
        gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    # Canny
    edges = cv2.Canny(gray, params["canny_low"], params["canny_high"])
    # Morphology
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (params["morph_k"], params["morph_k"]))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                              kernel, iterations=params["morph_iter"])
    # Contours
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < params["min_area"] or area > params["max_area"]:
            continue
        x, y, ww, hh = cv2.boundingRect(cnt)
        # pad
        pad = int(max(2, 0.02 * max(w, h)))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w-1, x + ww + pad)
        y1 = min(h-1, y + hh + pad)
        # compute simple heuristics for label
        aspect = ww / (hh + 1e-8)
        mean_intensity = int(np.mean(gray[y0:y1, x0:x1])) if (
            y1 > y0 and x1 > x0) else 255
        label = "candidate"
        score = None
        # heuristics - tune to your needs
        if aspect > params["aspect_crack_thresh"]:
            label = "crack"
            score = min(1.0, area / (params["max_area"] + 1e-6))
        elif mean_intensity < params["dark_thresh"] and area > params["delam_area_thresh"]:
            label = "corrosion"
            score = min(1.0, area / (params["max_area"] + 1e-6))
        elif area > params["delam_area_thresh"]:
            label = "delamination"
            score = min(1.0, area / (params["max_area"] + 1e-6))
        candidates.append(
            {"box": (x0, y0, x1, y1), "area": area, "label": label, "score": score})
    return closed, candidates


# --- Session state for stored image ---
if "stored_image_bytes" not in st.session_state:
    st.session_state.stored_image_bytes = None

# --- UI: left column for capture/upload, right column for controls+results ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Capture / Upload")
    st.markdown(
        "Take a snapshot with your webcam, or upload a file. The image will be stored for live tuning.")
    cam = st.camera_input("Take a photo (allow camera access)")
    uploaded = st.file_uploader("Or upload an image", type=[
                                "jpg", "jpeg", "png", "bmp"])
    if st.button("Clear stored image"):
        st.session_state.stored_image_bytes = None
        st.success("Stored image cleared")

    # store whichever is most recent (camera has priority)
    if cam is not None:
        # camera_input returns UploadedFile-like
        st.session_state.stored_image_bytes = cam.getvalue()
    elif uploaded is not None:
        st.session_state.stored_image_bytes = uploaded.getvalue()

    if st.session_state.stored_image_bytes is None:
        st.info("No stored image yet. Capture or upload one to start tuning.")
    else:
        # show stored image preview
        try:
            img = Image.open(io.BytesIO(st.session_state.stored_image_bytes))
            st.image(img, caption="Stored original image",
                     use_column_width=True)
        except Exception as e:
            st.error(f"Could not open stored image: {e}")

with col2:
    st.subheader("Live Image Parameters Tuning (changes apply instantly)")
    # Processing parameters
    clahe_clip = st.slider("CLAHE clip limit", 1.0, 8.0, 2.0, 0.1)
    clahe_tile = st.selectbox("CLAHE tile grid", [4, 8, 16], index=1)
    blur_k = st.selectbox("Gaussian blur kernel (odd)",
                          [1, 3, 5, 7, 9], index=1)
    canny_low = st.slider("Canny low threshold", 1, 255, 50)
    canny_high = st.slider("Canny high threshold", 1, 500, 150)
    morph_k = st.selectbox("Morph kernel size", [3, 5, 7, 9], index=0)
    morph_iter = st.slider("Morph close iterations", 0, 5, 1)
    min_area = st.number_input(
        "Min contour area (px)", min_value=1, max_value=100000, value=50, step=1)
    max_area = st.number_input(
        "Max contour area (px)", min_value=100, max_value=10000000, value=8000, step=100)
    aspect_crack_thresh = st.slider(
        "Aspect ratio threshold for 'crack' (w/h)", 1.5, 50.0, 6.0, 0.5)
    dark_thresh = st.slider(
        "Mean-intensity threshold (dark) for corrosion heuristic (0-255)", 0, 255, 60)
    delam_area_thresh = st.number_input(
        "Area threshold for delamination heuristic (px)", min_value=10, max_value=100000, value=1500, step=10)
    show_labels = st.checkbox("Show labels on boxes", value=True)
    show_edge_map = st.checkbox("Show edge map (debug)", value=False)
    download_name = st.text_input(
        "Download file name", value="ashmi_annotated.jpg")

    # Build params dict
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

    # If we have a stored image, process it automatically
    if st.session_state.stored_image_bytes is not None:
        try:
            pil_img = Image.open(io.BytesIO(
                st.session_state.stored_image_bytes)).convert("RGB")
            bgr = pil_to_bgr(pil_img)
            # optional resize for very large images (keeps responsiveness)
            max_side = 1600
            h, w = bgr.shape[:2]
            if max(h, w) > max_side:
                scale = max_side / float(max(h, w))
                bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)),
                                 interpolation=cv2.INTER_AREA)

            # detection & annotation (runs every time any widget changes)
            edge_map, candidates = detect_candidates(bgr, params)
            annotated = annotate_image(
                bgr, candidates, show_labels=show_labels)

            # Layout: show original, annotated, and optionally edge map
            st.markdown("### Results")
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(pil_img, caption="Original (stored)",
                         use_column_width=True)
            with col_b:
                st.image(bgr_to_pil(annotated),
                         caption="Annotated (live)", use_column_width=True)

            if show_edge_map:
                st.markdown("Edge map (after CLAHE -> Canny -> Morph close)")
                st.image(edge_map, use_column_width=True)

            # Download annotated image
            buf = io.BytesIO()
            pil_out = bgr_to_pil(annotated)
            pil_out.save(buf, format="JPEG")
            buf.seek(0)
            st.download_button("Download annotated image", data=buf,
                               file_name=download_name, mime="image/jpeg")
            # Also show detection summary table
            if len(candidates) > 0:
                st.markdown(
                    f"**Detected {len(candidates)} candidate(s)** — first few:")
                # show small list
                for i, c in enumerate(candidates[:8]):
                    lbl = c.get("label", "candidate")
                    area = int(c.get("area", 0))
                    st.write(f"- {i+1}: {lbl}, area={area}px, box={c['box']}")
            else:
                st.info("No candidates detected with current parameters.")
        except Exception as e:
            st.error(f"Processing failed: {e}")
    # If no stored image
    else:
        st.info(
            "No image stored. Capture with the camera or upload an image in the left panel.")

# Footer tips
st.markdown("---")
st.markdown("""
**Tips**
- Take a photo under even diffuse lighting for best results.
- Use `Canny low/high` to control sensitivity. Lower = more edges (more false positives).
- Increase `min area` to ignore speckle/noise.
- `Aspect ratio` is useful to identify long thin cracks.
""")
