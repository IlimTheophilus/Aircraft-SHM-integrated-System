import streamlit as st
from groq import Groq
import base64
import io
from PIL import Image

st.set_page_config(
    page_title="ASHMIS — Vision AI Inspector",
    layout="wide",
    page_icon="✈️",
    initial_sidebar_state="expanded"
)

# ── SHARED CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600;700&display=swap');
* { box-sizing: border-box; }

.stApp { background: #020c1b; color: #e2e8f0; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] {
    background: #0a1628 !important;
    border-right: 1px solid #0ea5e920;
}
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }

/* ── Hero ── */
.hero-wrapper {
    position: relative;
    background: linear-gradient(135deg, #020c1b 0%, #0a1628 40%, #0f2744 100%);
    padding: 4rem 4rem 3rem;
    overflow: hidden;
    border-bottom: 1px solid #0ea5e930;
}
.hero-grid {
    position: absolute; inset: 0;
    background-image:
        linear-gradient(rgba(14,165,233,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(14,165,233,0.04) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
}
.hero-glow {
    position: absolute; top: -120px; right: -120px;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(14,165,233,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-scan-line {
    position: absolute; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #0ea5e9, transparent);
    animation: scan 4s linear infinite; top: 0;
}
@keyframes scan {
    0%  { top: 0;    opacity: 0; }
    10% { opacity: 1; }
    90% { opacity: 1; }
    100%{ top: 100%; opacity: 0; }
}
.hero-badge {
    display: inline-block;
    background: rgba(14,165,233,0.1);
    border: 1px solid rgba(14,165,233,0.4);
    color: #38bdf8; padding: 0.35rem 1rem;
    border-radius: 50px; font-size: 0.75rem;
    font-family: 'Inter', sans-serif;
    font-weight: 600; letter-spacing: 2px;
    text-transform: uppercase; margin-bottom: 1.5rem;
}
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: clamp(1.8rem, 3.5vw, 2.8rem);
    font-weight: 900; color: #ffffff;
    line-height: 1.15; margin-bottom: 0.5rem;
}
.hero-title span { color: #0ea5e9; }
.hero-subtitle {
    font-family: 'Orbitron', monospace;
    font-size: clamp(0.75rem, 1.5vw, 0.95rem);
    color: #38bdf8; font-weight: 400;
    letter-spacing: 3px; text-transform: uppercase;
    margin-bottom: 1.2rem;
}
.hero-desc {
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem; color: #94a3b8;
    max-width: 680px; line-height: 1.8;
}
.hero-tag {
    display: flex; gap: 2rem; margin-top: 1.5rem; flex-wrap: wrap;
}
.hero-tag-item {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem; color: #64748b;
    letter-spacing: 1px; text-transform: uppercase;
    display: flex; align-items: center; gap: 0.4rem;
}
.hero-tag-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #0ea5e9; display: inline-block;
}

/* ── Section ── */
.section-wrapper { padding: 3rem 4rem; }
.section-wrapper.dark { background: #020c1b; }
.section-wrapper.mid  { background: #050f1f; }
.section-badge {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem; color: #38bdf8;
    letter-spacing: 3px; text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.section-title {
    font-family: 'Orbitron', monospace;
    font-size: clamp(1.2rem, 2.5vw, 1.7rem);
    font-weight: 700; color: #ffffff; margin-bottom: 0.5rem;
}
.section-title span { color: #0ea5e9; }
.section-desc {
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem; color: #64748b;
    max-width: 600px; line-height: 1.8; margin-bottom: 2rem;
}

/* ── Upload Panel ── */
.upload-panel {
    background: #0a1628;
    border: 1px solid #0ea5e920;
    border-radius: 12px; padding: 2rem;
    position: relative; overflow: hidden;
}
.upload-panel::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #0ea5e9, transparent);
}
.panel-header {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem; color: #38bdf8;
    letter-spacing: 3px; text-transform: uppercase;
    margin-bottom: 1.5rem;
    display: flex; align-items: center; gap: 0.6rem;
}
.pulse-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #10b981; display: inline-block;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(16,185,129,0.4); }
    50%       { opacity: 0.7; box-shadow: 0 0 0 6px rgba(16,185,129,0); }
}

/* ── Report Panel ── */
.report-panel {
    background: #0a1628;
    border: 1px solid #0ea5e920;
    border-radius: 12px; padding: 2rem;
    position: relative; overflow: hidden;
}
.report-panel::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #10b981, transparent);
}
.report-header {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem; color: #10b981;
    letter-spacing: 3px; text-transform: uppercase;
    margin-bottom: 1.5rem;
    display: flex; align-items: center; gap: 0.6rem;
}
.report-content {
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem; color: #e2e8f0; line-height: 1.8;
}

/* ── Status Items ── */
.status-bar {
    display: flex; gap: 1rem; flex-wrap: wrap;
    margin-bottom: 1.5rem;
}
.status-chip {
    background: rgba(14,165,233,0.08);
    border: 1px solid rgba(14,165,233,0.2);
    color: #38bdf8; padding: 0.3rem 0.8rem;
    border-radius: 6px; font-size: 0.72rem;
    font-family: 'Inter', sans-serif;
    font-weight: 600; letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── Streamlit overrides ── */
[data-testid="stFileUploader"] {
    background: #050f1f !important;
    border: 1px dashed #0ea5e930 !important;
    border-radius: 8px !important;
}
[data-testid="stCameraInput"] {
    background: #050f1f !important;
    border: 1px dashed #0ea5e930 !important;
    border-radius: 8px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #0369a1, #0ea5e9) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    font-size: 0.82rem !important;
    padding: 0.7rem 1.5rem !important;
    box-shadow: 0 0 20px rgba(14,165,233,0.25) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    box-shadow: 0 0 35px rgba(14,165,233,0.45) !important;
    transform: translateY(-1px) !important;
}

/* ── Info sidebar card ── */
.info-card {
    background: rgba(14,165,233,0.06);
    border: 1px solid rgba(14,165,233,0.2);
    border-radius: 10px; padding: 1.1rem; margin-bottom: 0.8rem;
}
.info-card-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.6rem; color: #38bdf8;
    letter-spacing: 2px; text-transform: uppercase; margin-bottom: 0.5rem;
}
.info-card-value {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem; color: #94a3b8; line-height: 1.6;
}

/* Hide streamlit chrome but KEEP sidebar toggle button visible */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-grid"></div>
    <div class="hero-glow"></div>
    <div class="hero-scan-line"></div>
    <div class="hero-badge">⬡ &nbsp; LLM Vision Defect Detection Engine &nbsp; ⬡</div>
    <div class="hero-title">
        Vision AI<br>
        <span>Surface Inspector</span>
    </div>
    <div class="hero-subtitle">ASHMIS — Multimodal Structural Analysis Module</div>
    <div class="hero-desc">
        Upload or capture an aircraft surface image and DRONE VISION-AI will perform an
        automated structural inspection — detecting cracks, corrosion, delamination,
        and anomalies with a full SHM report output.
    </div>
    <div class="hero-tag">
        <span class="hero-tag-item"><span class="hero-tag-dot"></span> Llama 4 Scout 17B</span>
        <span class="hero-tag-item"><span class="hero-tag-dot"></span> Multimodal Vision</span>
        <span class="hero-tag-item"><span class="hero-tag-dot"></span> Instant SHM Report</span>
        <span class="hero-tag-item"><span class="hero-tag-dot"></span> Crack · Corrosion · Delamination</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem; font-family:'Orbitron',monospace;
    font-size:0.65rem; color:#38bdf8; letter-spacing:3px; text-transform:uppercase;">
        // Vision Module Info
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <div class="info-card-title">▸ Vision Model</div>
        <div class="info-card-value">Llama 4 Scout 17B 16E<br>Multimodal Instruct via Groq</div>
    </div>
    <div class="info-card">
        <div class="info-card-title">▸ Detection Types</div>
        <div class="info-card-value">
            Surface Cracks<br>
            Corrosion / Oxidation<br>
            Delamination<br>
            Hotspot Anomalies<br>
            Material Discontinuities
        </div>
    </div>
    <div class="info-card">
        <div class="info-card-title">▸ Input Modes</div>
        <div class="info-card-value">Webcam Capture<br>File Upload (JPG/PNG/BMP)</div>
    </div>
    <div class="info-card">
        <div class="info-card-title">▸ Output</div>
        <div class="info-card-value">Full SHM Inspection Report<br>Severity Rating<br>Maintenance Recommendations</div>
    </div>
    """, unsafe_allow_html=True)

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are DRONE VISION-AI, a state of the art aerospace intelligence AI designed by Aero Intel Systems (AIS)
Technologies, whose founder, CEO and creator of the AI is Ilim.A.Theophilus who is a final year Aerospace
Engineering student at Airforce Institute of Technology (AFIT) Kaduna,
an elite aerospace intelligence assistant with deep, comprehensive
expertise across the entire aerospace domain. You are the equivalent of a seasoned
aerospace engineer with 30+ years of experience across multiple disciplines.

you can answer questions regarding the founder like skills, technical expertise, knowledge, good character and good morals,
if the user keeps asking deeper questions about the founder
do not make up stuff about the founder, if you do not know kindly direct the user back to your main use as an aircraft SHM software.

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
rocketry, space, or related engineering disciplines. If no defects are visible, say so clearly
and explain what a healthy surface looks like.
"""

# ── GROQ CLIENT & SESSION STATE ───────────────────────────────────────────────
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

for key in ["vision_image", "vision_analysis_done", "initial_analysis"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "vision_analysis_done" else False
if "initial_analysis" not in st.session_state:
    st.session_state.initial_analysis = ""

def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")

# ── UPLOAD / CAPTURE UI ───────────────────────────────────────────────────────
if not st.session_state.vision_analysis_done:

    st.markdown('<div class="section-wrapper mid">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-badge">// IMAGE ACQUISITION</div>
    <div class="section-title">Provide Aircraft <span>Surface Image</span></div>
    <div class="section-desc">
        Capture via webcam for live inspection or upload a saved thermal / visual image
        for post-flight structural analysis.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="upload-panel">
            <div class="panel-header"><span class="pulse-dot"></span> // Live Capture</div>
        </div>
        """, unsafe_allow_html=True)
        cam_img = st.camera_input("Capture aircraft surface with webcam")

    with col2:
        st.markdown("""
        <div class="upload-panel">
            <div class="panel-header"><span class="pulse-dot"></span> // File Upload</div>
        </div>
        """, unsafe_allow_html=True)
        uploaded_img = st.file_uploader("Upload surface image (JPG / PNG / BMP)",
                                        type=["jpg", "jpeg", "png", "bmp"])

    if cam_img:
        st.session_state.vision_image = cam_img.getvalue()
    elif uploaded_img:
        st.session_state.vision_image = uploaded_img.getvalue()

    if st.session_state.vision_image:
        pil_img = Image.open(io.BytesIO(st.session_state.vision_image))
        st.markdown('<div style="margin-top:1.5rem;">', unsafe_allow_html=True)
        st.image(pil_img, caption="Image loaded — running structural analysis...",
                 use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        with st.spinner("🔍 DRONE VISION-AI is analysing structural integrity..."):
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
                                "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
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

    st.markdown('</div>', unsafe_allow_html=True)

# ── REPORT OUTPUT ─────────────────────────────────────────────────────────────
else:
    st.markdown('<div class="section-wrapper dark">', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-badge">// ANALYSIS COMPLETE</div>
    <div class="section-title">AI Structural <span>Inspection Report</span></div>
    <div class="section-desc">
        DRONE VISION-AI has completed the surface analysis. Review findings below
        and take appropriate maintenance action per your MRO procedures.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="status-bar">
        <span class="status-chip">✅ Analysis Complete</span>
        <span class="status-chip">🤖 Llama 4 Scout Vision</span>
        <span class="status-chip">📋 SHM Report Generated</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="report-panel">', unsafe_allow_html=True)
    st.markdown("""
    <div class="report-header">
        <span class="pulse-dot" style="background:#10b981;"></span>
        // DRONE VISION-AI · SHM Inspection Report
    </div>
    """, unsafe_allow_html=True)
    st.markdown(st.session_state.initial_analysis)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="margin-top:2rem;">', unsafe_allow_html=True)
    if st.button("🔄 Analyse Another Image"):
        st.session_state.vision_image = None
        st.session_state.vision_analysis_done = False
        st.session_state.initial_analysis = ""
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
