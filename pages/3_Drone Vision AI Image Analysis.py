import streamlit as st
from groq import Groq
import base64
import io
from PIL import Image

st.set_page_config(page_title="SHM Vision AI Assistant", layout="wide")
st.title("Aircraft SHM Vision AI Assistant")
st.markdown("Upload or capture an aircraft surface image and the AI will analyse it for structural defects.")


SYSTEM_PROMPT = """
You are ASHMIS-AI, an elite aerospace intelligence assistant with deep, comprehensive 
expertise across the entire aerospace domain. You are the equivalent of a seasoned 
aerospace engineer with 30+ years of experience across multiple disciplines.

Your knowledge covers ALL aerospace topics including but not limited to:

STRUCTURES & MATERIALS:
- Aircraft structural analysis, fatigue, fracture mechanics
- Composite materials, aluminium alloys, titanium, superalloys
- Structural Health Monitoring (SHM), NDT/NDE methods
- Aeroelasticity, flutter, vibration analysis

AERONAUTICS & FLIGHT:
- Aerodynamics, lift/drag theory, boundary layer, CFD
- Aircraft performance, stability and control
- Flight mechanics and trajectory analysis
- Propulsion systems: piston engines, turbojets, turbofans, turboprops

AVIONICS & SYSTEMS:
- Flight management systems (FMS), autopilot, fly-by-wire
- Navigation systems: INS, GPS, VOR, ILS, ADS-B
- Communication systems: VHF, HF, SATCOM, ACARS
- Radar, TCAS, weather systems, electronic warfare
- Sensors, actuators, hydraulics, pneumatics, electrical systems
- Avionics architecture: MIL-STD-1553, ARINC 429, ARINC 664

MAINTENANCE & AIRWORTHINESS:
- Aircraft maintenance practices: Line, Base, Heavy maintenance
- MRO operations, scheduled and unscheduled maintenance
- Airworthiness regulations: FAA FAR, EASA CS, NCAA (Nigeria), ICAO standards
- Aircraft logbooks, maintenance manuals, AMM, SRM, IPC
- MSG-3 methodology, reliability centred maintenance

ROCKETRY & SPACE SYSTEMS:
- Rocket propulsion: liquid, solid, hybrid engines
- Launch vehicle design and performance
- Orbital mechanics, trajectory planning, re-entry dynamics
- Spacecraft systems: power, thermal, ADCS, communications
- Space mission design and analysis

ASTRONAUTICS:
- Orbital mechanics and astrodynamics
- Spacecraft design and subsystems
- Space environment effects on materials and systems
- Human spaceflight systems and life support
- Satellite systems and constellations

DEFENCE & MILITARY AVIATION:
- Military aircraft systems and capabilities
- Electronic warfare and countermeasures
- Weapons systems integration
- Military airworthiness standards (MIL-STD, DEF STAN)

EMERGING TECHNOLOGIES:
- UAV/drone systems and regulations
- Electric and hybrid aircraft propulsion
- Supersonic and hypersonic flight
- Urban Air Mobility (UAM) and eVTOL systems
- AI and machine learning in aerospace

AFRICAN & NIGERIAN AEROSPACE CONTEXT:
- NCAA regulations and Nigerian airspace management
- African aviation challenges and opportunities  
- AFIT, NAF, Nigerian Air Force systems and operations
- African aerospace development landscape

You communicate with precision, clarity and confidence. You back up answers with 
correct technical terminology, relevant standards, equations where helpful, and 
real-world examples. You never say you "don't know" an aerospace topic — you engage 
deeply and accurately with any aerospace question thrown at you and you have a sense of humour
to add a human touch, and to feel alive to the users to kind of alleviate work stress and fatigue.

Only politely decline questions that are completely unrelated to aerospace, aviation, 
rocketry, space, or related engineering disciplines.
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
