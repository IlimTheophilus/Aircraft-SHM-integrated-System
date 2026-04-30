import streamlit as st
import google.generativeai as genai
from PIL import Image
import io

st.set_page_config(page_title="SHM Vision AI Assistant", layout="wide")
st.title("🔍 Aircraft SHM Vision AI Assistant")
st.markdown("Upload or capture an aircraft surface image — the AI will analyse it for structural defects and answer your questions about it.")

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

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=SYSTEM_PROMPT
)

# --- Image Input ---
st.subheader("1. Provide an Aircraft Surface Image")
col1, col2 = st.columns(2)

with col1:
    cam_img = st.camera_input("Capture with webcam")
with col2:
    uploaded_img = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png", "bmp"])

# Store image in session
if cam_img:
    st.session_state.vision_image = cam_img.getvalue()
elif uploaded_img:
    st.session_state.vision_image = uploaded_img.getvalue()

# Display stored image
if "vision_image" in st.session_state and st.session_state.vision_image:
    pil_img = Image.open(io.BytesIO(st.session_state.vision_image))
    st.image(pil_img, caption="Image loaded for analysis", use_container_width=True)

    # Auto-analyse on first load
    if "vision_analysis_done" not in st.session_state:
        st.session_state.vision_analysis_done = False
    
    if not st.session_state.vision_analysis_done:
        with st.spinner("🔍 Running structural defect analysis..."):
            response = model.generate_content([
                "Analyse this aircraft surface image for structural defects. Provide a detailed SHM report.",
                pil_img
            ])
            st.session_state.initial_analysis = response.text
            st.session_state.vision_analysis_done = True

    # Show analysis report
    st.markdown("### 📋 AI Structural Analysis Report")
    st.markdown(st.session_state.get("initial_analysis", ""))

    # --- Follow-up Chat ---
    st.markdown("---")
    st.subheader("2. Ask Follow-up Questions About This Image")

    if "vision_chat_history" not in st.session_state:
        st.session_state.vision_chat_history = []
        st.session_state.vision_gemini_chat = model.start_chat(history=[])

    for msg in st.session_state.vision_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask a specific question about this image...")

    if user_q:
        st.session_state.vision_chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Include image context in follow-up
                full_prompt = f"Referring to the aircraft surface image previously analysed:\n\n{user_q}"
                response = st.session_state.vision_gemini_chat.send_message(full_prompt)
                reply = response.text
                st.markdown(reply)

        st.session_state.vision_chat_history.append({"role": "assistant", "content": reply})

    # Reset button
    if st.button("🔄 Clear image and start new analysis"):
        for key in ["vision_image", "vision_analysis_done", "initial_analysis", 
                    "vision_chat_history", "vision_gemini_chat"]:
            st.session_state.pop(key, None)
        st.rerun()
else:
    st.info("No image loaded yet. Capture or upload an aircraft surface image above to begin analysis.")
