import streamlit as st
from groq import Groq
import base64
import io
from PIL import Image

st.set_page_config(page_title="SHM Vision AI Assistant", layout="wide")
st.title("Aircraft SHM Vision AI Assistant")
st.markdown("Upload or capture an aircraft surface image and the AI will analyse it for structural defects.")

SYSTEM_PROMPT = """
You are an expert Aircraft Structural Health Monitoring (SHM) engineer and
computer vision specialist. When given an image of an aircraft surface or component:
1. Carefully analyse the image for any visible structural defects
2. Identify and describe: cracks, corrosion, delamination, fatigue damage,
   surface anomalies, paint damage, or any other structural concerns
3. Estimate the severity (low / medium / high / critical)
4. Recommend appropriate maintenance action
5. Reference relevant NDT standards where applicable (MIL-HDBK, ASTM, etc.)

Be precise, safety-focused, and use correct aerospace terminology.
If no defects are visible, say so clearly and explain what a healthy surface looks like.
"""

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")

# Reset state if no image
if "vision_image" not in st.session_state:
    st.session_state.vision_image = None
if "vision_analysis_done" not in st.session_state:
    st.session_state.vision_analysis_done = False
if "initial_analysis" not in st.session_state:
    st.session_state.initial_analysis = ""

# --- If no analysis done yet, show upload UI ---
if not st.session_state.vision_analysis_done:
    st.subheader("Provide an Aircraft Surface Image")
    col1, col2 = st.columns(2)

    with col1:
        cam_img = st.camera_input("Capture with webcam")
    with col2:
        uploaded_img = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png", "bmp"])

    if cam_img:
        st.session_state.vision_image = cam_img.getvalue()
    elif uploaded_img:
        st.session_state.vision_image = uploaded_img.getvalue()

    if st.session_state.vision_image:
        pil_img = Image.open(io.BytesIO(st.session_state.vision_image))
        st.image(pil_img, caption="Image loaded for analysis", use_container_width=True)

        with st.spinner("Running structural defect analysis..."):
            b64_image = encode_image(st.session_state.vision_image)

            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}"
                                }
                            },
                            {
                                "type": "text",
                                "text": "Analyse this aircraft surface image for structural defects. Provide a detailed SHM inspection report."
                            }
                        ]
                    }
                ],
                max_tokens=1024
            )
            st.session_state.initial_analysis = response.choices[0].message.content
            st.session_state.vision_analysis_done = True
            st.rerun()

# --- After analysis, show report and prompt for new image ---
else:
    st.markdown("### AI Structural Analysis Report")
    st.markdown(st.session_state.initial_analysis)

    st.markdown("---")
    st.info("Analysis complete. Click below to analyse another image.")

    if st.button("Analyse Another Image"):
        st.session_state.vision_image = None
        st.session_state.vision_analysis_done = False
        st.session_state.initial_analysis = ""
        st.rerun()
