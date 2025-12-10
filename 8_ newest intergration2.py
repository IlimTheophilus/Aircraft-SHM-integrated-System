
# app_with_live_and_agent.py
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageEnhance
import time
import io
import threading
import openai  # optional; only used when API key present

# ---------------------------
# Page config + CSS (futuristic)
# ---------------------------
st.set_page_config(
    page_title="Structure Automator - Live + Agent", layout="wide")
st.markdown("""
<style>
body { background-color: #0b1020; color: #dbefff; font-family: Inter, sans-serif; }
.card { background: #0f1724; border-radius: 12px; padding: 14px; border: 1px solid rgba(255,255,255,0.03); }
.neon { color: #00e5ff; text-shadow: 0 0 10px rgba(0,229,255,0.12); font-weight:700; font-size:22px; }
.small { color: #9fb6d8; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Utility functions
# ---------------------------


@st.cache_resource
def load_model(path=None):
    # Default uses segmentation-capable weights if present
    try:
        if path:
            return YOLO(path)
        return YOLO("yolov8n-seg.pt")
    except Exception:
        return YOLO("yolov8n.pt")


def pil_to_cv2(img_pil):
    arr = np.array(img_pil.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv2_to_bytes(img_cv2):
    _, buf = cv2.imencode('.png', img_cv2)
    return buf.tobytes()


def apply_enhancements(img, brightness=1.0, contrast=1.0, denoise=0):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil = ImageEnhance.Brightness(pil).enhance(brightness)
    pil = ImageEnhance.Contrast(pil).enhance(contrast)
    img_out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    if denoise > 0:
        img_out = cv2.fastNlMeansDenoisingColored(
            img_out, None, denoise, denoise, 7, 21)
    return img_out


def draw_detections(img, results, conf_thresh=0.25, min_area=50, mask_opacity=0.35):
    out = img.copy()
    dets = []
    part_classes = ['wing', 'fuselage', 'empennage', 'landing_gear']
    defect_classes = ['crack', 'delamination', 'corrosion', 'impact']
    for r in results:
        for box in getattr(r, "boxes", []):
            conf = float(box.conf.cpu().numpy())
            if conf < conf_thresh:
                continue
            xyxy = box.xyxy.cpu().numpy().astype(int)[0]
            x1, y1, x2, y2 = xyxy
            w, h = x2-x1, y2-y1
            area = w*h
            if area < min_area:
                continue
            try:
                cls_idx = int(box.cls.cpu().numpy())
                cls_name = r.names[cls_idx]
            except Exception:
                cls_name = f'class_{int(box.cls.cpu().numpy())}'
            # color palette neon
            if cls_name in part_classes:
                color = (29, 233, 182)  # mint
            elif cls_name in defect_classes:
                color = (255, 195, 77)  # amber
            else:
                color = (255, 85, 127)  # candidate
                cls_name = 'candidate'
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"{cls_name} {conf:.2f}", (x1, max(
                y1-8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # segmentation mask if available
            try:
                if hasattr(box, 'mask') and box.mask is not None:
                    mask = box.mask.data.cpu().numpy()[0]
                    mask_resized = cv2.resize(mask, (w, h))
                    colored = np.zeros_like(out[y1:y2, x1:x2])
                    colored[mask_resized > 0.5] = color
                    out[y1:y2, x1:x2] = cv2.addWeighted(
                        out[y1:y2, x1:x2], 1-mask_opacity, colored, mask_opacity, 0)
            except Exception:
                pass
            dets.append({'class': cls_name, 'conf': conf,
                        'bbox': (x1, y1, x2, y2), 'area': area})
    return out, dets

# Simple offline explanation generator


def local_explain(defect_summary, selected_part):
    # defect_summary: list of {'class','conf','area'}
    if not defect_summary:
        return ("No defects detected. Suggestion: ensure camera angle, improve lighting, "
                "or decrease detection threshold to increase sensitivity.")
    lines = []
    lines.append(f"Analysis summary for *{selected_part}* (automated):")
    for d in defect_summary:
        cls = d['class']
        conf = d['conf']
        area = d['area']
        if cls == 'crack':
            lines.append(
                f"- Crack detected (conf {conf:.2f}, area {area}px): Immediate visual and ultrasonic inspection recommended. Cracks propagate under load — restrict operations until confirmed.")
        elif cls == 'delamination':
            lines.append(
                f"- Delamination-like signature (conf {conf:.2f}): schedule ultrasonic or tap testing and monitor growth. Consider patch repair based on depth.")
        elif cls == 'corrosion':
            lines.append(
                f"- Corrosion sign (conf {conf:.2f}): clean coating, inspect for pitting, and treat with inhibitor. Structural integrity check recommended for deep corrosion.")
        elif cls == 'impact':
            lines.append(
                f"- Impact damage candidate (conf {conf:.2f}): perform local structural assessment; look for subsurface damage.")
        elif cls == 'candidate':
            lines.append(
                f"- Candidate anomaly (conf {conf:.2f}): image quality or perspective might be insufficient; re-scan at higher resolution.")
        else:
            lines.append(
                f"- {cls} (conf {conf:.2f}): manual review recommended.")
    lines.append("\nSuggested next steps:\n1) Acquire higher resolution image or thermal fusion.\n2) Run NDT (ultrasonic / eddy current) on flagged zones.\n3) Log to SHM database and increase monitoring frequency.")
    return "\n".join(lines)

# Optional: OpenAI-powered explanation (requires API key)


def openai_explain(defect_summary, selected_part, openai_api_key):
    openai.api_key = openai_api_key
    prompt = "You are an aerospace structural health monitoring expert. Provide a concise, practical explanation and recommended next steps for the following automated detections on an aircraft part.\n\n"
    prompt += f"Aircraft part: {selected_part}\n\nDetections:\n"
    for d in defect_summary:
        prompt += f"- {d['class']} (confidence {d['conf']:.2f}, area {d['area']} px)\n"
    prompt += "\nProvide:\n1) short plain-language explanation of the likely physical issue\n2) recommended immediate actions (safety-critical first)\n3) suggested NDT methods and monitoring cadence\n4) estimation of severity (low/medium/high) with justification.\n"
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # change if needed; user can modify
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.25,
        )
        return resp['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"OpenAI API call failed: {e}\nFalling back to local explanation.\n\n" + local_explain(defect_summary, selected_part)


# ---------------------------
# App UI: Sidebar controls
# ---------------------------
st.sidebar.title("Structure Automator — Live & Agent")
model_path = st.sidebar.text_input(
    "YOLO model path (optional, leave blank for default)", "")
model = load_model(model_path if model_path.strip() else None)

st.sidebar.markdown("### Input / Mode")
mode = st.sidebar.radio("Mode", ["Image Analyze", "Live Video (Webcam/RTSP)"])
selected_part = st.sidebar.selectbox("Confirm Part (helps explanation):", [
                                     "wing", "fuselage", "empennage", "landing_gear", "other"])

st.sidebar.markdown("### Image Enhancements & Detection")
brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.0)
contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0)
denoise = st.sidebar.slider("Denoise level", 0, 20, 0)
conf_thresh = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.35)
min_area = st.sidebar.number_input(
    "Min detection area (px)", min_value=0, value=100, step=50)
mask_opacity = st.sidebar.slider("Mask opacity", 0.0, 0.8, 0.35)

st.sidebar.markdown("---")
st.sidebar.markdown("### ChatGPT-style Explanation")
openai_key = st.sidebar.text_input(
    "OpenAI API Key (optional)", type="password")
use_openai = st.sidebar.checkbox(
    "Use OpenAI to generate explanations (optional)", value=False)
if use_openai and not openai_key:
    st.sidebar.warning("Provide API key to enable OpenAI explanations.")

# Session storage for gallery & video state
if 'gallery' not in st.session_state:
    st.session_state['gallery'] = []
if 'video_running' not in st.session_state:
    st.session_state['video_running'] = False

# ---------------------------
# Main area
# ---------------------------
st.markdown("<div class='neon'>STRUCTURE AUTOMATOR — LIVE ANALYTICS</div>",
            unsafe_allow_html=True)
st.markdown("<div class='small'>Upload images, run live feeds, and ask the agent for explanations.</div>",
            unsafe_allow_html=True)
st.write("")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Input & Detection")

    if mode == "Image Analyze":
        uploaded = st.file_uploader(
            "Upload image (jpg/png/tiff)", type=["jpg", "png", "jpeg", "tiff"])
        cam_capture = None
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            cv_img = pil_to_cv2(image)
            enhanced = apply_enhancements(
                cv_img, brightness, contrast, denoise)
            # YOLO inference
            with st.spinner("Running YOLO..."):
                results = model(enhanced, conf=conf_thresh)
            annotated, detections = draw_detections(
                enhanced, results, conf_thresh, min_area, mask_opacity)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     caption="Annotated detection", use_column_width=True)
            st.markdown("**Detections**")
            for d in detections:
                st.write(
                    f"- {d['class']} (conf {d['conf']:.2f}, area {d['area']})")
            # store gallery
            st.session_state['gallery'].append({
                'time': int(time.time()),
                'part': selected_part,
                'detections': detections,
                'image': cv2_to_bytes(annotated)
            })
            # Explanation
            st.markdown("### Explanation (Agent)")
            if use_openai and openai_key:
                expl = openai_explain(detections, selected_part, openai_key)
                st.markdown(expl)
            else:
                st.markdown(local_explain(detections, selected_part))
    else:
        # Live video mode
        st.markdown("**Live Video Feed Detection**")
        st.markdown(
            "Use webcam (default) or provide RTSP / IP camera URL below.")
        rtsp_url = st.text_input(
            "RTSP / camera URL (leave blank for local webcam 0)")
        start = st.button("Start Live Feed")
        stop = st.button("Stop Live Feed")
        placeholder = st.empty()

        def run_video_stream(url):
            st.session_state['video_running'] = True
            cap = cv2.VideoCapture(0 if not url else url)
            last_frame_time = 0
            try:
                while st.session_state['video_running'] and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # throttle to ~10-15 fps
                    now = time.time()
                    if now - last_frame_time < 0.08:
                        time.sleep(0.02)
                        continue
                    last_frame_time = now
                    enhanced = apply_enhancements(
                        frame, brightness, contrast, denoise)
                    results = model(enhanced, conf=conf_thresh)
                    annotated, detections = draw_detections(
                        enhanced, results, conf_thresh, min_area, mask_opacity)
                    # display
                    placeholder.image(cv2.cvtColor(
                        annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
                    # store last frame detection summary in session_state for inspector
                    st.session_state['last_live'] = {'time': int(
                        time.time()), 'detections': detections, 'frame': cv2_to_bytes(annotated)}
                    # small sleep to yield
                    time.sleep(0.01)
            finally:
                try:
                    cap.release()
                except Exception:
                    pass
                st.session_state['video_running'] = False

        if start and (not st.session_state['video_running']):
            # start stream in a thread to avoid blocking main Streamlit thread
            t = threading.Thread(target=run_video_stream,
                                 args=(rtsp_url,), daemon=True)
            t.start()
        if stop and st.session_state['video_running']:
            st.session_state['video_running'] = False
            placeholder.empty()
        if 'last_live' in st.session_state:
            st.markdown("**Last live detection:**")
            last = st.session_state['last_live']
            if last.get('detections'):
                for d in last['detections']:
                    st.write(
                        f"- {d['class']} (conf {d['conf']:.2f}, area {d['area']})")
            if last.get('frame'):
                st.image(Image.open(io.BytesIO(
                    last['frame'])), use_column_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Inspector / Gallery")
    # Session gallery preview
    if st.session_state['gallery']:
        st.write(f"Saved scans: {len(st.session_state['gallery'])}")
        for idx, item in enumerate(reversed(st.session_state['gallery'][-6:])):
            st.image(Image.open(io.BytesIO(
                item['image'])), caption=f"{item['part']} — {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(item['time']))}")
            if st.button(f"Explain #{idx}", key=f"explain_{idx}"):
                # open explanation for that item
                dets = item['detections']
                if use_openai and openai_key:
                    st.markdown(openai_explain(dets, item['part'], openai_key))
                else:
                    st.markdown(local_explain(dets, item['part']))
    else:
        st.info(
            "No saved images yet. Analyze an image or run the live feed to populate gallery.")

    st.markdown("---")
    st.subheader("Agent Playground")
    prompt = st.text_area(
        "Ask the agent about the most recent detection (or type custom question)", height=120)
    if st.button("Get Agent Answer"):
        # If user uses OpenAI and provided key, call API; otherwise use local template fallback
        last = None
        if st.session_state.get('gallery'):
            last = st.session_state['gallery'][-1]
        elif st.session_state.get('last_live'):
            last = st.session_state['last_live']
        if not last:
            st.warning(
                "No detection available to explain yet. Run an analysis first.")
        else:
            dets = last.get('detections', [])
            if use_openai and openai_key:
                # build a factual context + user prompt
                context = "Automated detections:\n"
                for d in dets:
                    context += f"- {d['class']} (conf {d['conf']:.2f}, area {d['area']})\n"
                full_prompt = context + "\nUser question: " + \
                    (prompt if prompt.strip(
                    ) else "Explain the current detections and recommended next steps.")
                try:
                    openai.api_key = openai_key
                    resp = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": full_prompt}],
                        max_tokens=300,
                        temperature=0.2
                    )
                    st.markdown(resp['choices'][0]['message']['content'])
                except Exception as e:
                    st.error(
                        f"OpenAI call failed: {e}. Falling back to local explanation.")
                    st.markdown(local_explain(
                        dets, last.get('part', 'unknown')))
            else:
                # local explanation using simple heuristics + custom question handling (limited)
                st.markdown(local_explain(dets, last.get('part', 'unknown')))
    st.markdown("</div>", unsafe_allow_html=True)

# Footer notes
st.markdown("<div class='small'>Developer notes: For production, export the YOLO model to TensorRT/NVidia Jetson for high-throughput video inference. The OpenAI integration requires a valid API key and may incur costs.</div>", unsafe_allow_html=True)
