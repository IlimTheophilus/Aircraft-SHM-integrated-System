# ashmi_interactive_variantA.py
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io

# scikit-image imports
from skimage import exposure, filters, feature, morphology, measure, color, util

st.set_page_config(page_title="ASHMIS Interactive - Variant A", layout="wide")
st.title("ASHMIS Structural Defect Detection (SDD) Dashboard — Snapshot + Live Parameter Tuning (Variant A)")

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


def pil_to_array(pil_img: Image.Image):
    """Return an HxWx3 uint8 numpy array (RGB)."""
    return np.array(pil_img.convert("RGB"))


def array_to_pil(rgb_arr: np.ndarray):
    """Convert HxWx3 uint8 RGB numpy array to PIL Image."""
    return Image.fromarray(rgb_arr.astype(np.uint8))


def draw_annotations_on_array(rgb_arr: np.ndarray, candidates, show_labels=True):
    """Draw rectangles and labels on an RGB numpy array using PIL."""
    pil = array_to_pil(rgb_arr)
    draw = ImageDraw.Draw(pil)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for c in candidates:
        x0, y0, x1, y1 = c["box"]

        # Draw bounding box
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)

        if show_labels:
            lbl = c.get("label", "candidate")
            if c.get("score") is not None:
                lbl = f"{lbl} {c['score']:.2f}"

            # -------------------------------
            # FIXED: Replace textsize() → textbbox()
            # -------------------------------
            if font:
                bbox = draw.textbbox((0, 0), lbl, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            else:
                # fallback if font fails
                text_w = len(lbl) * 6
                text_h = 10

            text_x = x0
            text_y = max(0, y0 - text_h - 3)

            # background rectangle
            draw.rectangle(
                [text_x, text_y, text_x + text_w + 4, text_y + text_h + 2],
                fill=(0, 0, 0)
            )

            # label text
            draw.text((text_x + 2, text_y + 1), lbl,
                      fill=(255, 255, 255), font=font)

    return np.array(pil)


def detect_candidates(rgb_arr: np.ndarray, params):
    """
    Input: rgb_arr (HxWx3 uint8)
    Returns: edge_map (PIL-friendly array or boolean array), candidates list
    """
    # Convert to grayscale float (0..1)
    gray = color.rgb2gray(rgb_arr)  # float64 [0,1]

    # CLAHE / adaptive histogram equalization
    # skimage exposure.equalize_adapthist expects clip_limit and kernel_size (tile size)
    try:
        clahe = exposure.equalize_adapthist(gray, clip_limit=max(0.01, params["clahe_clip"]/8.0),
                                            kernel_size=(params["clahe_tile"], params["clahe_tile"]))
    except TypeError:
        # fallback if kernel_size not supported in this skimage version
        clahe = exposure.equalize_adapthist(
            gray, clip_limit=max(0.01, params["clahe_clip"]/8.0))

    # Gaussian blur (if kernel>1)
    blur_k = params["blur_k"]
    if blur_k > 1:
        sigma = max(0.3, blur_k / 2.0)
        blurred = filters.gaussian(clahe, sigma=sigma)
    else:
        blurred = clahe

    # Canny - thresholds need to be normalized to [0,1] (we assume slider used 0-255)
    low_t = np.clip(params["canny_low"] / 255.0, 0.0, 1.0)
    high_t = np.clip(params["canny_high"] / 255.0, 0.0, 1.0)
    # skimage.feature.canny has low_threshold and high_threshold arguments in [0,1]
    try:
        edges = feature.canny(
            blurred, low_threshold=low_t, high_threshold=high_t)
    except TypeError:
        # older versions may not accept both thresholds; fall back to sigma-based
        edges = feature.canny(blurred, sigma=1.0)

    # Morphological closing
    morph_k = max(1, int(params["morph_k"]))
    selem = morphology.square(morph_k)
    closed = edges.copy()
    for _ in range(max(1, params["morph_iter"])):
        closed = morphology.closing(closed, selem)

    # Label connected regions and compute properties
    label_img = measure.label(closed)
    regions = measure.regionprops(label_img)

    h, w = gray.shape
    candidates = []
    for region in regions:
        area = region.area
        if area < params["min_area"] or area > params["max_area"]:
            continue
        minr, minc, maxr, maxc = region.bbox  # rows(y), cols(x)
        # pad
        pad = int(max(2, 0.02 * max(w, h)))
        x0 = max(0, minc - pad)
        y0 = max(0, minr - pad)
        x1 = min(w-1, maxc + pad)
        y1 = min(h-1, maxr + pad)
        # compute heuristics
        ww = max(1, (x1 - x0))
        hh = max(1, (y1 - y0))
        aspect = ww / (hh + 1e-8)
        # mean intensity from CLAHE result (0..1) convert to 0..255
        mean_intensity = int(
            np.mean(clahe[y0:y1, x0:x1]) * 255) if (y1 > y0 and x1 > x0) else 255
        label = "candidate"
        score = None
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

    # Prepare edge map for display: convert boolean to uint8 grayscale 0..255
    edge_map_disp = (closed.astype(np.uint8) * 255)
    return edge_map_disp, candidates


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
            rgb_arr = pil_to_array(pil_img)
            # optional resize for very large images (keeps responsiveness)
            max_side = 1600
            h, w = rgb_arr.shape[:2]
            if max(h, w) > max_side:
                scale = max_side / float(max(h, w))
                new_w = int(w * scale)
                new_h = int(h * scale)
                pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
                rgb_arr = pil_to_array(pil_img)

            # detection & annotation (runs every time any widget changes)
            edge_map, candidates = detect_candidates(rgb_arr, params)
            annotated_arr = draw_annotations_on_array(
                rgb_arr, candidates, show_labels=show_labels)

            # Layout: show original, annotated, and optionally edge map
            st.markdown("### Results")
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(pil_img, caption="Original (stored)",
                         use_column_width=True)
            with col_b:
                st.image(array_to_pil(annotated_arr),
                         caption="Annotated (live)", use_column_width=True)

            if show_edge_map:
                st.markdown("Edge map (after CLAHE -> Canny -> Morph close)")
                # edge_map is uint8 0..255
                st.image(Image.fromarray(edge_map), use_column_width=True)

            # Download annotated image
            buf = io.BytesIO()
            pil_out = array_to_pil(annotated_arr)
            pil_out.save(buf, format="JPEG")
            buf.seek(0)
            st.download_button("Download annotated image", data=buf,
                               file_name=download_name, mime="image/jpeg")

            # Also show detection summary table
            if len(candidates) > 0:
                st.markdown(
                    f"**Detected {len(candidates)} candidate(s)** — first few:")
                for i, c in enumerate(candidates[:8]):
                    lbl = c.get("label", "candidate")
                    area = int(c.get("area", 0))
                    st.write(f"- {i+1}: {lbl}, area={area}px, box={c['box']}")
            else:
                st.info("No candidates detected with current parameters.")
        except Exception as e:
            st.error(f"Processing failed: {e}")
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
