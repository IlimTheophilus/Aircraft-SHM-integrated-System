import streamlit as st
from groq import Groq

st.set_page_config(
    page_title="ASHMIS — Aerospace AI Chatbot",
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
    max-width: 680px; line-height: 1.8; margin-bottom: 0;
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

/* ── Chat Container ── */
.chat-wrapper {
    padding: 2rem 4rem;
    background: #020c1b;
}
.chat-panel {
    background: #0a1628;
    border: 1px solid #0ea5e920;
    border-radius: 12px;
    overflow: hidden;
}
.chat-panel-header {
    background: #050f1f;
    border-bottom: 1px solid #0ea5e920;
    padding: 1rem 1.5rem;
    display: flex; align-items: center; gap: 0.8rem;
}
.chat-panel-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.7rem; color: #38bdf8;
    letter-spacing: 3px; text-transform: uppercase;
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

/* ── Info Chips ── */
.chip-row {
    display: flex; gap: 0.7rem; flex-wrap: wrap;
    padding: 1.5rem 4rem 0;
}
.chip {
    background: rgba(14,165,233,0.07);
    border: 1px solid rgba(14,165,233,0.2);
    color: #38bdf8; padding: 0.3rem 0.9rem;
    border-radius: 50px; font-size: 0.72rem;
    font-family: 'Inter', sans-serif;
    font-weight: 600; letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── Streamlit chat override ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stChatMessageContent"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    color: #e2e8f0 !important;
    line-height: 1.75 !important;
}
[data-testid="stChatInput"] textarea {
    background: #0a1628 !important;
    border: 1px solid #0ea5e930 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    border-radius: 8px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #0ea5e9 !important;
    box-shadow: 0 0 0 2px rgba(14,165,233,0.15) !important;
}

/* ── Sidebar AI card ── */
.ai-card {
    background: rgba(14,165,233,0.06);
    border: 1px solid rgba(14,165,233,0.2);
    border-radius: 10px; padding: 1.2rem;
    margin: 1rem;
}
.ai-card-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.62rem; color: #38bdf8;
    letter-spacing: 2px; text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.ai-card-value {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem; color: #94a3b8; line-height: 1.6;
}

/* Hide streamlit chrome but KEEP sidebar toggle button visible */
#MainMenu, footer { visibility: hidden; }
header { visibility: hidden; }
[data-testid="collapsedControl"] { visibility: visible !important; display: flex !important; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are DRONE VISION-AI, a state of the art aerospace intelligence AI designed by Aero Intel Systems (AIS)
Technologies whose founder, CEO and creator of the AI is Ilim.A.Theophilus
who is a final year Aerospace Engineering student at Airforce Institute of Technology (AFIT) Kaduna,
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
rocketry, space, or related engineering disciplines. If no defects are visible, say so clearly and explain what a healthy surface looks like.
"""

# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-grid"></div>
    <div class="hero-glow"></div>
    <div class="hero-scan-line"></div>
    <div class="hero-badge">⬡ &nbsp; AI Aerospace Intelligence Engine &nbsp; ⬡</div>
    <div class="hero-title">
        DRONE VISION-AI<br>
        <span>Aerospace Chatbot</span>
    </div>
    <div class="hero-subtitle">ASHMIS — Intelligent Flight Operations Assistant</div>
    <div class="hero-desc">
        A Llama 3.3 70B powered aerospace intelligence assistant with 30+ years equivalent
        expertise across SHM, avionics, rocketry, military systems, and Nigerian aviation
        regulations. Ask anything aerospace — from fatigue cracks to orbital mechanics.
    </div>
    <div class="hero-tag">
        <span class="hero-tag-item"><span class="hero-tag-dot"></span> Llama 3.3 70B</span>
        <span class="hero-tag-item"><span class="hero-tag-dot"></span> NCAA / EASA / ICAO</span>
        <span class="hero-tag-item"><span class="hero-tag-dot"></span> SHM Specialist</span>
        <span class="hero-tag-item"><span class="hero-tag-dot"></span> NDT / MRO Expert</span>
        <span class="hero-tag-item"><span class="hero-tag-dot"></span> AIS Technologies</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── TOPIC CHIPS ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="chip-row">
    <span class="chip">🔩 Structures & Fatigue</span>
    <span class="chip">🌡️ NDT / NDE</span>
    <span class="chip">✈️ Aerodynamics</span>
    <span class="chip">⚙️ Propulsion</span>
    <span class="chip">📡 Avionics</span>
    <span class="chip">🛸 Rocketry</span>
    <span class="chip">🛡️ Military Systems</span>
    <span class="chip">🌍 Nigerian Aviation</span>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem; font-family:'Orbitron',monospace;
    font-size:0.65rem; color:#38bdf8; letter-spacing:3px; text-transform:uppercase;">
        // AI System Info
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="ai-card">
        <div class="ai-card-title">▸ Model</div>
        <div class="ai-card-value">Llama 3.3 70B Versatile<br>via Groq Inference</div>
    </div>
    <div class="ai-card">
        <div class="ai-card-title">▸ Designed By</div>
        <div class="ai-card-value">Ilim A. Theophilus<br>Aerospace Engineering, AFIT Kaduna<br>Founder, AIS Technologies</div>
    </div>
    <div class="ai-card">
        <div class="ai-card-title">▸ Expertise Coverage</div>
        <div class="ai-card-value">
            Structures · Materials · SHM<br>
            Aerodynamics · Propulsion<br>
            Avionics · Navigation<br>
            Rocketry · Astronautics<br>
            Military Aviation · UAV<br>
            NCAA · EASA · FAA · ICAO
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)

    if st.button("🗑 Clear Chat History", use_container_width=True):
        st.session_state.chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        st.rerun()

# ── GROQ CLIENT ───────────────────────────────────────────────────────────────
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]

# ── CHAT AREA ─────────────────────────────────────────────────────────────────
st.markdown('<div style="padding: 2rem 4rem 0;">', unsafe_allow_html=True)

st.markdown("""
<div class="chat-panel">
    <div class="chat-panel-header">
        <span class="pulse-dot"></span>
        <span class="chat-panel-title">// DRONE VISION-AI · Aerospace Intelligence Engine · Online</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Render chat history
for msg in st.session_state.chat_history:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

st.markdown('</div>', unsafe_allow_html=True)

# ── INPUT ─────────────────────────────────────────────────────────────────────
st.markdown('<div style="padding: 0 4rem 3rem;">', unsafe_allow_html=True)

user_prompt = st.chat_input("Ask about SHM, defects, NDT, avionics, rocketry, regulations...")

if user_prompt:
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analysing..."):
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=st.session_state.chat_history,
                max_tokens=1024
            )
            reply = response.choices[0].message.content
            st.markdown(reply)

    st.session_state.chat_history.append({"role": "assistant", "content": reply})

st.markdown('</div>', unsafe_allow_html=True)
