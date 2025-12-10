import streamlit as st
from PIL import Image

st.title(" Take a Picture of the Aircraft Part to be Analysed")

st.subheader("For Best Results: ")
st.text(" 1. Ensure the Picture is taken in a well lighted enviroment and all"
        " detachable components are removed from the airframe to avoid obstruction or interference.")
st.text(" 2. Ensure to keep the Camera Steady for Accurate Surface Analysis")

img_file_buffer = st.camera_input("  ")

if img_file_buffer is not none:
    image = image.open(img_file_buffer)
    st.image(image, caption="captured image", use_container_width=True)

if st.button(" Upload Image for Analysis"):
    st.write(" Your Image has been submitted for Analysis")
