import streamlit as st
from PIL import Image


# Title
st.title(" Airframe Structural Health Monitoring Integrated System (ASHMIS) ")


# Image


image_path = "images/ASHMIS img.png"
img = Image.open(image_path)

# Fit image to the column width
st.image(img, caption="", use_container_width=True)



# Text

# Header
st.subheader(" A Low-Cost AI Integrated System for Aircraft Maintenance ")

# Write
st.write(""" Welcome to ASHMIS, an Aircraft Structural Health Monitoring (SHM) Application designed to use Thermal/Visual 
         Imaging Technology and Artificial Intelligence to carry out Non-Destructive Testing and Inspections on the Aircraft's 
         Airframe (Focused on the Wings for now) capable of detecting early-stage cracks and anomalies in Aircraft Structures 
         using affordable sensors and machine learning techniques.
         """)


# Image


image_path = "images/fighter jet pic.jpg"
img = Image.open(image_path)

# Fit image to the column width
st.image(img, caption="", use_container_width=True)


st.write("""ASHMIS is an integrated, lowcost Airframe Structural Health Monitoring System that leverages computer vision, thermal
         imaging and Machine learning to automatically detect structural defects in aircraft components such as wings, fuselage and 
         tail sections. This module enables the acquisition, processing, and analysis of visual and thermal images to identify potential 
         issues such as cracks, delamination, and corrosion, facilitating timely maintenance and ensuring flight safety. Using a hybrid 
         detection architecture, combining thermal gradient analysis, pixel level defect segmentation and ML-driven pattern recognition, 
         ASHMIS evaluates each snapshot for indications of structural degradation, hotspots, material discontinuities, or other early-stage
         failutre signatures.""")


# --- HEADER SECTION ---
st.markdown("""
    <style>
    .hero {
        text-align: center;
        padding: 50px 20px;
        background: linear-gradient(135deg,#0369a1, #0ea5e9);
        color: yellow;
        border-radius: 10px;
    }
    .hero h1 {
        font-size: 3em;
        font-weight: 800;
    }
    .hero p {
        font-size: 1.3em;
        color: #e0f2fe;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>Drone Vision AI</h1>
    <p>Real-time Aircraft Wing Health Monitoring and Defect Detection powered by Artificial Intelligence </p>
</div>
""", unsafe_allow_html=True)


st.write(" Aircraft's Structural Integrity is vital to flight safety and maintenance cost efficiency. Conventional inspection methods although effective, are expensive and labour intensive, limiting their frequent use in developing Aerospace enviroments such as Nigeria. This System Integrates Thermal imaging and Visual "
         "Imaging to detect cracks, delamination and other structural defects.")


st.write(" Such an Application is designed to demonstrate that affordable imaging and AI technologies can be integrated to acheive reliable structural health assessment.")

# Simple input
name = st.text_input(
    "Enter your Name Below to join the team, and be part of something special to come:")

# Button
if st.button("submit name"):
    st.write(
        f"Hello, {name}! Your name has been submitted, standby for processing 😄")

# Slider example
age = st.slider(
    " Enter your years of experience in Python Programming", 0, 30, 2)
st.write(
    f"wow you have {age} years of experience. After having {age} years experience in writing, give me a summary of the time you spent Programming")


# Image

image_path = "images/picture of plane1.webp"
image = Image.open(image_path)

# Fit image to the column width
st.image(image, caption="", use_container_width=True)


# --- END SECTION ---
st.markdown("""
    <style>
    .zero {
        text-align: center;
        padding: 50px 100px;
        background: linear-gradient(135deg, #0ea5e9, #0369a1);
        color: red;
        border-radius: 10px;
        margin-bottom: 50px;
    }
    .hero h1 {
        font-size: 3em;
        font-weight: 800;
    }
    .hero p {
        font-size: 1.3em;
        color: #e0f2fe;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1> Thanks for Visiting my Web Application</h1>
    <p> Programmed by Ilim.A.Theophilus</p>
            <p> Creator of ASHMIS </p>
</div>
""", unsafe_allow_html=True)
