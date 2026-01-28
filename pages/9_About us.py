import streamlit as st
from PIL import Image


# Title
st.title(" Aerospace Intelligent Systems (AIS) Technologies ")


# Image


image_path = "images/Aero Intel img.png"
img = Image.open(image_path)

# Fit image to the column width
st.image(img, caption="", use_container_width=True)



# Text

# Header
st.header(" AeroIntel Systems (AIS) Technologies ")

st.subheader(" Building the Next Generation of Intelligence Systems ")

# Write
st.write("""Aerospace Intelligent Systems (AIS) Technologies is an Aerospace and Defense technology company dedicated to the design and development of intelligent, mission-critical systems for aviation, space,
and national defense applications.
We build advanced aerospace systems that integrate engineering, sensing, data intelligence, and autonomy to support aircraft,systems platforms, and defense infrastructure across their full operational lifecycle. 
Our focus is not a single product or domain, but the creation of scalable aerospace and defense capabilities that strengthen operational effectiveness, technological independence, and strategic readiness.
          Aerospace Intelligent Systems (AIS) Technologies recognizes aerospace technology as a strategic National asset.
Our work contributes to:

•  Enhanced Aircraft Maintenance and platform readiness

•  Reduced dependence on foreign diagnostic and maintenance systems

•  Stronger National Defense capabilities

•  Low-Cost Indigenous Aerospace technology development

By building local expertise and scalable systems, we aim to support National defense readiness while competing globally.
________________________________________

         """)


# Image


image_path = "images/fighter jet pic.jpg"
img = Image.open(image_path)

# Fit image to the column width
st.image(img, caption="", use_container_width=True)


st.subheader(" Our Vision ")

st.write("""To become a globally respected aerospace and defense systems company delivering intelligent technologies that enhance airspace safety, mission effectiveness, and national security,
while enabling Nigeria as an emerging Aerospace nation to build and control it’s own critical capabilities.
________________________________________
""")

st.subheader(" Our Mission ")

st.write("""To design, develop, and deploy advanced aerospace and defense systems that combine:

•  Aerospace engineering

•  Intelligent sensing

•  Data-driven decision systems

•  Artificial Intelligence Autonomous and adaptive technologies

into reliable, scalable solutions for civil aviation, defense, and strategic aerospace applications.
________________________________________

""")



st.subheader(" Our First Platform: Aircraft Structural Health Monitoring and Integrated System (ASHMIS) ")


# Image

image_path = "images/ASHMIS img.png"
img = Image.open(image_path)

# Fit image to the column width
st.image(img, caption="", use_container_width=True)



st.write("""ASHMIS is a web-based aerospace intelligence platform designed to transform aircraft maintenance and inspection.
Using thermal visual imaging and computer vision, ASHMIS scans aircraft surfaces to detect:

•  Structural cracks

•  Delamination

• Corrosion

• Material fatigue and hidden defects

""")


st.write("""ASHMIS turns visual and thermal data into actionable engineering insight, enabling:

•  Faster inspections

•  Early fault detection

•  Reduced Aircraft on Ground (AOG) time

•  Improved airworthiness decision-making

This is not just software, it is a foundation for intelligent Aircraft systems.
________________________________________

""")


st.subheader(" Why It Matters ")

st.text("""In many regions, including Africa, aircraft maintenance still relies heavily on: 

•  Manual inspections
•  Time-based maintenance schedules
•  Limited access to advanced diagnostic tools
     
 AeroIntel Systems Technologies aims to change that reality by building technologies that:
•  Increase aviation safety
•  Reduce operational costs
•  Strengthen local Aerospace capability
•  Support airlines, MROs, defense operators, and regulators
Our work contributes directly to safer skies, stronger infrastructure, and a more self-reliant Aerospace ecosystem.
________________________________________

""")


# Image

image_path = "images/global-air-travel-connectivity.jpg"
image = Image.open(image_path)

# Fit image to the column width
st.image(image, caption="", use_container_width=True)



st.subheader(" Beyond Maintenance ")

st.write("""While ASHMIS begins with structural health monitoring, AeroIntel Systems Technologies is not limited to maintenance alone.
Our long-term roadmap includes:

•  Intelligent aircraft systems

•  Aerospace data platforms

•  Defense and surveillance technologies

•  Autonomous inspection and monitoring tools

ASHMIS is the first step, not the destination.
AeroIntel Systems (AIS) Technologies is built with a systems-level mindset, combining engineering rigor, software intelligence, and real operational needs.
We are building today what the aerospace industry will rely on tomorrow.
Built for Scale. Designed for Impact.

""")



# Image


image_path = "images/airplane-night-runway.jpg"
img = Image.open(image_path)

# Fit image to the column width
st.image(img, caption="", use_container_width=True)
