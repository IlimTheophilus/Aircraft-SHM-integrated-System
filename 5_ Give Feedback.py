import streamlit as st
from PIL import Image

# Simple feedback input
name = st.text_input(
    " How was your Experience using the application")

st.button(" Send Feedback ")

st.slider(" Rate yor Experience on a Scale of 1-10", 0, 10)


st.set_page_config(page_title="Drone Health Monitor", layout="wide")

st.title("🛸 Drone Health Monitoring System")
st.markdown("Upload a drone image for structural analysis.")

col1, col2 = st.columns([1, 2])
with col1:
    st.image("drone1.jpg", caption="ProForce Tactical Drone",
             use_container_width=True)
with col2:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded", use_container_width=True)
        st.success("✅ Image uploaded successfully")
