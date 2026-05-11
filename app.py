
import streamlit as st
from PIL import Image
import os
st.set_page_config(
    page_title="ASHMIS — Aircraft Structural Health Monitoring",
    layout="wide",
    page_icon="✈️",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600;700&display=swap');
* { box-sizing: border-box; }
/* ── Global ── */
.stApp { background: #020c1b; color: #e2e8f0; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] {
    background: #0a1628 !important;
    border-right: 1px solid #0ea5e920;
}
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] p { color: #94a3b8 !important; }
/* ── Hero ── */
.hero-wrapper {
    position: relative;
    background: linear-gradient(135deg, #020c1b 0%, #0a1628 40%, #0f2744 100%);
    padding: 5rem 4rem 4rem;
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
.hero-glow-left {
    position: absolute; bottom: -100px; left: -100px;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(6,182,212,0.08) 0%, transparent 70%);
    pointer-events: none;
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
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 900; color: #ffffff;
    line-height: 1.15; margin-bottom: 0.5rem;
    letter-spacing: 1px;
}
.hero-title span { color: #0ea5e9; }
.hero-subtitle {
    font-family: 'Orbitron', monospace;
    font-size: clamp(0.9rem, 2vw, 1.1rem);
    color: #38bdf8; font-weight: 400;
    letter-spacing: 3px; text-transform: uppercase;
    margin-bottom: 1.5rem;
}
.hero-desc {
    font-family: 'Inter', sans-serif;
    font-size: 1rem; color: #94a3b8;
    max-width: 680px; line-height: 1.8;
    margin-bottom: 2.5rem;
}
.hero-buttons { display: flex; gap: 1rem; flex-wrap: wrap; }
.btn-primary {
    background: linear-gradient(135deg, #0369a1, #0ea5e9);
    color: white; padding: 0.8rem 2rem;
    border-radius: 8px; font-weight: 700;
    font-size: 0.9rem; text-decoration: none;
    font-family: 'Inter', sans-serif;
    border: none; letter-spacing: 1px;
    text-transform: uppercase;
    box-shadow: 0 0 30px rgba(14,165,233,0.3);
}
.btn-outline {
    background: transparent;
    border: 1px solid rgba(14,165,233,0.5);
    color: #38bdf8; padding: 0.8rem 2rem;
    border-radius: 8px; font-weight: 600;
    font-size: 0.9rem; text-decoration: none;
    font-family: 'Inter', sans-serif;
    letter-spacing: 1px; text-transform: uppercase;
}
.hero-scan-line {
    position: absolute; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #0ea5e9, transparent);
    animation: scan 4s linear infinite; top: 0;
}
@keyframes scan {
    0%  { top: 0;    opacity: 0; }
    10% { opacity: 1; }
    90% { opacity: 1; }
    100%{ top: 100%; opacity: 0; }
}
.hero-tag {
    display: flex; gap: 2rem; margin-top: 2rem; flex-wrap: wrap;
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
/* ── Stats Bar ── */
.stats-bar {
    background: #0a1628;
    border-top: 1px solid #0ea5e915;
    border-bottom: 1px solid #0ea5e915;
    padding: 1.8rem 4rem;
    display: flex; justify-content: space-around;
    flex-wrap: wrap; gap: 1rem;
}
.stat-item { text-align: center; }
.stat-value {
    font-family: 'Orbitron', monospace;
    font-size: 2rem; font-weight: 700; color: #0ea5e9;
}
.stat-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem; color: #64748b;
    letter-spacing: 2px; text-transform: uppercase;
    margin-top: 0.2rem;
}
/* ── Section Wrapper ── */
.section-wrapper { padding: 4rem; }
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
    font-size: clamp(1.4rem, 3vw, 2rem);
    font-weight: 700; color: #ffffff;
    margin-bottom: 0.5rem;
}
.section-title span { color: #0ea5e9; }
.section-desc {
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem; color: #64748b;
    max-width: 600px; line-height: 1.8;
    margin-bottom: 3rem;
}
/* ── Feature Cards ── */
.features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.5rem; margin-top: 2rem;
}
.feature-card {
    background: #0a1628;
    border: 1px solid #0ea5e915;
    border-radius: 12px; padding: 1.8rem;
    position: relative; overflow: hidden;
    transition: border-color 0.3s, transform 0.2s;
}
.feature-card::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #0ea5e9, transparent);
}
.feature-icon {
    font-size: 2.2rem; margin-bottom: 1rem;
}
.feature-title {
    font-family: 'Inter', sans-serif;
    font-size: 1rem; font-weight: 700;
    color: #e2e8f0; margin-bottom: 0.6rem;
}
.feature-desc {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem; color: #64748b; line-height: 1.7;
}
.feature-tag {
    display: inline-block; margin-top: 1rem;
    background: rgba(14,165,233,0.1);
    border: 1px solid rgba(14,165,233,0.2);
    color: #38bdf8; padding: 0.2rem 0.7rem;
    border-radius: 4px; font-size: 0.68rem;
    font-family: 'Inter', sans-serif;
    letter-spacing: 1px; text-transform: uppercase;
}
/* ── How It Works ── */
.how-steps {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 0; margin-top: 2rem; position: relative;
}
.how-step {
    background: #0a1628;
    border: 1px solid #0ea5e915;
    padding: 2rem 1.5rem; text-align: center;
    position: relative;
}
.how-step::after {
    content: '→';
    position: absolute; right: -14px; top: 50%;
    transform: translateY(-50%);
    color: #0ea5e9; font-size: 1.2rem; z-index: 2;
}
.how-step:last-child::after { display: none; }
.step-number {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem; color: #0ea5e9;
    letter-spacing: 2px; text-transform: uppercase;
    margin-bottom: 1rem;
}
.step-icon { font-size: 2rem; margin-bottom: 0.8rem; }
.step-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem; font-weight: 700; color: #e2e8f0;
    margin-bottom: 0.5rem;
}
.step-desc {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem; color: #64748b; line-height: 1.6;
}
/* ── System Status ── */
.status-panel {
    background: #0a1628;
    border: 1px solid #0ea5e920;
    border-radius: 12px; padding: 2rem;
    font-family: 'Inter', sans-serif;
}
.status-header {
    font-family: 'Orbitron', monospace;
    font-size: 0.7rem; color: #38bdf8;
    letter-spacing: 3px; text-transform: uppercase;
    margin-bottom: 1.5rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.pulse-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #10b981; display: inline-block;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(16,185,129,0.4); }
    50% { opacity: 0.7; box-shadow: 0 0 0 6px rgba(16,185,129,0); }
}
.status-item {
    display: flex; justify-content: space-between;
    align-items: center; padding: 0.7rem 0;
    border-bottom: 1px solid #0ea5e910;
    font-size: 0.85rem;
}
.status-item:last-child { border-bottom: none; }
.status-name { color: #94a3b8; }
.status-ok   { color: #10b981; font-weight: 600; font-size: 0.75rem; letter-spacing: 1px; }
.status-warn { color: #f59e0b; font-weight: 600; font-size: 0.75rem; letter-spacing: 1px; }
.status-pend { color: #f59e0b; font-weight: 600; font-size: 0.75rem; letter-spacing: 1px; }
/* ── Tech Stack ── */
.tech-grid {
    display: flex; flex-wrap: wrap;
    gap: 1rem; margin-top: 1.5rem;
}
.tech-badge {
    background: #0a1628;
    border: 1px solid #0ea5e920;
    color: #94a3b8; padding: 0.6rem 1.2rem;
    border-radius: 8px; font-size: 0.82rem;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
}
.tech-badge span { color: #38bdf8; margin-right: 0.4rem; }
/* ── Footer ── */
.footer {
    background: #020c1b;
    border-top: 1px solid #0ea5e915;
    padding: 3rem 4rem;
    display: flex; justify-content: space-between;
    align-items: center; flex-wrap: wrap; gap: 1.5rem;
}
.footer-brand {
    font-family: 'Orbitron', monospace;
    font-size: 1.1rem; font-weight: 700; color: #0ea5e9;
}
.footer-info {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem; color: #475569; margin-top: 0.3rem;
}
.footer-links {
    display: flex; gap: 2rem; flex-wrap: wrap;
}
.footer-link {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem; color: #64748b;
    text-decoration: none; letter-spacing: 1px;
    text-transform: uppercase;
}
/* ── Compliance Banner ── */
.compliance-bar {
    background: rgba(14,165,233,0.05);
    border: 1px solid rgba(14,165,233,0.15);
    border-radius: 10px; padding: 1.2rem 2rem;
    display: flex; align-items: center;
    gap: 1rem; flex-wrap: wrap; margin-top: 2rem;
}
.compliance-item {
    display: flex; align-items: center; gap: 0.4rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem; color: #64748b;
    letter-spacing: 1px; text-transform: uppercase;
    font-weight: 600;
}
.compliance-dot { color: #0ea5e9; font-size: 0.5rem; }
/* ── Info Cards ── */
.info-card {
    background: #0a1628;
    border: 1px solid #0ea5e915;
    border-radius: 10px; padding: 1.5rem;
}
.info-card-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.7rem; color: #38bdf8;
    letter-spacing: 2px; text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.info-card-value {
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem; color: #e2e8f0; line-height: 1.7;
}
/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }


/* FIX: hide sidebar nav WITHOUT breaking toggle button */
[data-testid="stSidebarNav"] {
    visibility: hidden;
    height: 0;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-grid"></div>
    <div class="hero-glow"></div>
    <div class="hero-glow-left"></div>
    <div class="hero-scan-line"></div>
    <div class="hero-badge">⬡ &nbsp; AI-Powered Aircraft Inspection System &nbsp; ⬡</div>
    <div class="hero-title">
        Airframe Structural<br>
        <span>Health Monitoring</span><br>
        Integrated System
    </div>
    <div class="hero-subtitle">ASHMIS — Next-Generation NDT Intelligence</div>
    <div class="hero-desc">
        A low-cost, AI-driven platform that fuses thermal and visual imaging with machine learning
        to detect cracks, corrosion, delamination and structural anomalies in aircraft components —
        enabling predictive maintenance and progressive degradation tracking in real time.
    </div>
    <div class="hero-tag">
        <span class="hero-tag-item"><span class="hero-tag-dot"></span> NCAA Compliant</span>
        <span class="hero-tag-item"><span class="hero-tag-dot"></span> EASA Part-145</span>
        <span class="hero-tag-item"><span class="hero-tag-dot"></span> ICAO Annex 6</span>
        <span class="hero-tag-item"><span class="hero-tag-dot"></span> NAF Standards</span>
        <span class="hero-tag-item"><span class="hero-tag-dot"></span> MIL-HDBK</span>
    </div>
</div>
""", unsafe_allow_html=True)
# ── STATS BAR ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stats-bar">
    <div class="stat-item">
        <div class="stat-value">3+</div>
        <div class="stat-label">Defect Types Detected</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">2</div>
        <div class="stat-label">Imaging Modalities</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">RT</div>
        <div class="stat-label">Real-Time Analysis</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">AI</div>
        <div class="stat-label">LLM Vision Engine</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">PDF</div>
        <div class="stat-label">Auto Report Generation</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">2</div>
        <div class="stat-label">Aviation Sectors</div>
    </div>
</div>
""", unsafe_allow_html=True)
# ── ABOUT SECTION ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-wrapper mid">', unsafe_allow_html=True)
col_left, col_right = st.columns([3, 2], gap="large")
with col_left:
    st.markdown("""
    <div class="section-badge">// ABOUT THE SYSTEM</div>
    <div class="section-title">What is <span>ASHMIS</span>?</div>
    <div class="section-desc">
        ASHMIS is an integrated, low-cost Airframe Structural Health Monitoring System that
        leverages computer vision, thermal imaging, and machine learning to automatically detect
        structural defects in aircraft components — wings, fuselage, tail sections, and more.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card" style="margin-bottom:1rem;">
        <div class="info-card-title">▸ Mission Statement</div>
        <div class="info-card-value">
            To demonstrate that affordable imaging and AI technologies can be integrated
            to achieve reliable structural health assessment, reducing inspection time and cost
            in developing aerospace environments such as Nigeria.
        </div>
    </div>
    <div class="info-card" style="margin-bottom:1rem;">
        <div class="info-card-title">▸ Detection Capability</div>
        <div class="info-card-value">
            Cracks &nbsp;•&nbsp; Corrosion &nbsp;•&nbsp; Delamination &nbsp;•&nbsp;
            Hotspots &nbsp;•&nbsp; Surface Anomalies &nbsp;•&nbsp; Material Discontinuities
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="compliance-bar">
        <div class="compliance-item"><span class="compliance-dot">●</span> NCAA Nig.CARs</div>
        <div class="compliance-item"><span class="compliance-dot">●</span> EASA Part-M</div>
        <div class="compliance-item"><span class="compliance-dot">●</span> EASA Part-145</div>
        <div class="compliance-item"><span class="compliance-dot">●</span> ICAO Annex 6</div>
        <div class="compliance-item"><span class="compliance-dot">●</span> MIL-HDBK-1823</div>
        <div class="compliance-item"><span class="compliance-dot">●</span> MIL-STD-1629</div>
    </div>
    """, unsafe_allow_html=True)
with col_right:
    try:
        img = Image.open("images/ASHMIS img.png")
        st.image(img, use_container_width=True)
    except:
        st.markdown("""
        <div style="background:#0a1628;border:1px solid #0ea5e920;border-radius:12px;
        padding:4rem 2rem;text-align:center;color:#0ea5e9;">
            <div style="font-size:4rem;">✈️</div>
            <div style="font-family:'Orbitron',monospace;font-size:0.8rem;
            letter-spacing:2px;margin-top:1rem;">ASHMIS</div>
        </div>
        """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
# ── FEATURES ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-wrapper dark">', unsafe_allow_html=True)
st.markdown("""
<div class="section-badge">// CORE CAPABILITIES</div>
<div class="section-title">System <span>Features</span></div>
<div class="section-desc">
    A full-suite NDT intelligence platform combining imaging technology,
    AI inference, and regulatory-grade documentation.
</div>
<div class="features-grid">
    <div class="feature-card">
        <div class="feature-icon">🔍</div>
        <div class="feature-title">AI-Driven Defect Detection</div>
        <div class="feature-desc">
            Computer vision pipeline using CLAHE preprocessing, Canny edge detection,
            morphological analysis, and AI inference to detect structural anomalies
            with bounding box annotation and confidence scoring.
        </div>
        <span class="feature-tag">Computer Vision</span>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🌡️</div>
        <div class="feature-title">Thermal & Visual Fusion</div>
        <div class="feature-desc">
            Dual-modality imaging combining infrared thermal camera data with
            visual RGB imaging for comprehensive structural assessment — detecting
            subsurface anomalies invisible to standard cameras.
        </div>
        <span class="feature-tag">Thermal Imaging</span>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🤖</div>
        <div class="feature-title">Vision AI Assistant</div>
        <div class="feature-desc">
            LLM-powered vision assistant that analyses uploaded aircraft surface images
            and generates detailed SHM inspection reports with severity ratings
            and maintenance recommendations.
        </div>
        <span class="feature-tag">Generative AI</span>
    </div>
    <div class="feature-card">
        <div class="feature-icon">📋</div>
        <div class="feature-title">Regulatory Inspection Reports</div>
        <div class="feature-desc">
            Auto-generate downloadable PDF maintenance inspection reports
            compliant with NCAA Nig.CARs, EASA Part-145, and NAF Technical
            Orders — with full signature blocks and certification.
        </div>
        <span class="feature-tag">NCAA • EASA • NAF</span>
    </div>
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <div class="feature-title">Maintenance Log & Tracking</div>
        <div class="feature-desc">
            Aviation-standard maintenance event logging per aircraft registration,
            flight hours, and inspection type — enabling progressive degradation
            tracking and airworthiness record keeping.
        </div>
        <span class="feature-tag">MRO Standard</span>
    </div>
    <div class="feature-card">
        <div class="feature-icon">💬</div>
        <div class="feature-title">Aerospace AI Chatbot</div>
        <div class="feature-desc">
            Full-spectrum aerospace intelligence assistant covering SHM, avionics,
            aerodynamics, rocketry, astronautics, military systems, and Nigerian
            aviation regulations. Powered by Llama 3.3 70B.
        </div>
        <span class="feature-tag">Llama 3.3 70B</span>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
# ── HOW IT WORKS ──────────────────────────────────────────────────────────────
