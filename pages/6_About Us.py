import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="ASHMIS — VELTHORIS Technologies",
    layout="wide",
    page_icon="✈️",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600;700&display=swap');
* { box-sizing: border-box; }

.stApp { background: #020c1b; color: #e2e8f0; }
.block-container { padding: 0 !important; }

section[data-testid="stSidebar"] {
    background: #0a1628 !important;
    border-right: 1px solid #0ea5e920;
}
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }

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
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(14,165,233,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-glow-left {
    position: absolute; bottom: -80px; left: -80px;
    width: 350px; height: 350px;
    background: radial-gradient(circle, rgba(6,182,212,0.07) 0%, transparent 70%);
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
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 900; color: #ffffff;
    line-height: 1.15; margin-bottom: 0.5rem;
}
.hero-title span { color: #0ea5e9; }
.hero-subtitle {
    font-family: 'Orbitron', monospace;
    font-size: clamp(0.85rem, 1.8vw, 1.05rem);
    color: #38bdf8; font-weight: 400;
    letter-spacing: 3px; text-transform: uppercase; margin-bottom: 1.5rem;
}
.hero-desc {
    font-family: 'Inter', sans-serif;
    font-size: 1rem; color: #94a3b8;
    max-width: 720px; line-height: 1.8;
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

/* ── Section ── */
.section-wrapper { padding: 4rem; }
.section-wrapper.dark { background: #020c1b; }
.section-wrapper.mid  { background: #050f1f; }
.section-badge {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem; color: #38bdf8;
    letter-spacing: 3px; text-transform: uppercase; margin-bottom: 0.8rem;
}
.section-title {
    font-family: 'Orbitron', monospace;
    font-size: clamp(1.4rem, 3vw, 2rem);
    font-weight: 700; color: #ffffff; margin-bottom: 0.5rem;
}
.section-title span { color: #0ea5e9; }
.section-desc {
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem; color: #64748b;
    max-width: 640px; line-height: 1.8; margin-bottom: 2.5rem;
}

/* ── Info Cards ── */
.info-card {
    background: #0a1628;
    border: 1px solid #0ea5e915;
    border-radius: 10px; padding: 1.6rem; margin-bottom: 1.2rem;
    position: relative; overflow: hidden;
}
.info-card::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #0ea5e9, transparent);
}
.info-card-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem; color: #38bdf8;
    letter-spacing: 2px; text-transform: uppercase; margin-bottom: 0.8rem;
}
.info-card-value {
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem; color: #e2e8f0; line-height: 1.8;
}

/* ── Bullet list cards ── */
.bullet-list {
    font-family: 'Inter', sans-serif;
    font-size: 0.88rem; color: #94a3b8;
    line-height: 2; list-style: none; padding: 0; margin: 0;
}
.bullet-list li::before { content: '▸ '; color: #0ea5e9; font-weight: 700; }

/* ── Feature grid ── */
.features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 1.2rem; margin-top: 1.5rem;
}
.feature-card {
    background: #0a1628;
    border: 1px solid #0ea5e915;
    border-radius: 12px; padding: 1.6rem;
    position: relative; overflow: hidden;
}
.feature-card::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #0ea5e9, transparent);
}
.feature-icon { font-size: 2rem; margin-bottom: 0.8rem; }
.feature-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.5rem;
}
.feature-desc {
    font-family: 'Inter', sans-serif;
    font-size: 0.83rem; color: #64748b; line-height: 1.7;
}

/* ── Image frame ── */
.img-frame {
    background: #0a1628;
    border: 1px solid #0ea5e920;
    border-radius: 12px; padding: 0.5rem;
    overflow: hidden; margin-bottom: 1.5rem;
}

/* ── Stats bar ── */
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
    font-size: 1.8rem; font-weight: 700; color: #0ea5e9;
}
.stat-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem; color: #64748b;
    letter-spacing: 2px; text-transform: uppercase; margin-top: 0.3rem;
}

/* ── Pulse dot ── */
.pulse-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #10b981; display: inline-block;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(16,185,129,0.4); }
    50%       { opacity: 0.7; box-shadow: 0 0 0 6px rgba(16,185,129,0); }
}

/* ── Roadmap items ── */
.roadmap-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem; margin-top: 1.5rem;
}
.roadmap-item {
    background: #0a1628;
    border: 1px solid #0ea5e910;
    border-radius: 10px; padding: 1.3rem;
    font-family: 'Inter', sans-serif;
}
.roadmap-num {
    font-family: 'Orbitron', monospace;
    font-size: 0.6rem; color: #0ea5e9;
    letter-spacing: 2px; text-transform: uppercase; margin-bottom: 0.5rem;
}
.roadmap-title {
    font-size: 0.88rem; font-weight: 700;
    color: #e2e8f0; margin-bottom: 0.3rem;
}
.roadmap-desc { font-size: 0.78rem; color: #64748b; line-height: 1.6; }

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

/* ═══════════════════════════════════════════════════
   SIDEBAR — PERMANENTLY VISIBLE
   ═══════════════════════════════════════════════════ */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }
.stDeployButton { display: none !important; }

[data-testid="collapsedControl"]            { display: none !important; }
[data-testid="stSidebarCollapseButton"]     { display: none !important; }
section[data-testid="stSidebar"]
  > div:first-child button                  { display: none !important; }

section[data-testid="stSidebar"] {
    transform: translateX(0) !important;
    min-width: 21rem !important;
    width: 21rem !important;
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
    <div class="hero-badge">⬡ &nbsp; Aerospace & Defense Technology Company &nbsp; ⬡</div>
    <div class="hero-title">
        <span> VELTHORIS </span><br>
        Technologies
    </div>
    <div class="hero-subtitle">BUILT FOR IMPACT</div>
    <div class="hero-desc">
        An Aerospace and Defense technology company dedicated to the design and development
        of intelligent, mission-critical systems for aviation, space, and national defense
        integrating engineering, sensing, data intelligence, and autonomy across the full
        operational lifecycle.
    </div>
    <div class="hero-tag">
        <span class="hero-tag-item"><span class="hero-tag-dot"></span> Aviation Systems</span>
        <span class="hero-tag-item"><span class="hero-tag-dot"></span> Defense Technology</span>
        <span class="hero-tag-item"><span class="hero-tag-dot"></span> Space Systems</span>
        <span class="hero-tag-item"><span class="hero-tag-dot"></span> Indigenous Innovation</span>
        <span class="hero-tag-item"><span class="hero-tag-dot"></span> AFIT Kaduna</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── STATS BAR ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stats-bar">
    <div class="stat-item">
        <div class="stat-value">1st</div>
        <div class="stat-label">Indigenous AI SHM Platform</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">AI</div>
        <div class="stat-label">Vision + Language Models</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">NDT</div>
        <div class="stat-label">Non-Destructive Testing</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">MRO</div>
        <div class="stat-label">Maintenance Operations</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">🌍</div>
        <div class="stat-label">Africa-First Strategy</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── ABOUT SECTION ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-wrapper mid">', unsafe_allow_html=True)

col_left, col_right = st.columns([3, 2], gap="large")
with col_left:
    st.markdown("""
    <div class="section-badge">// ABOUT VELTHORIS TECHNOLOGIES</div>
    <div class="section-title">Who We <span>Are</span></div>
    <div class="section-desc">
        VELTHORIS Technologies is a company that builds advanced aerospace systems that strengthen operational
        effectiveness, technological independence, and strategic readiness recognizing
        aerospace technology as a strategic National asset. It is the first Aerospace company in Nigeria with the CEO and Founder
        being Ilim A.Theophilus, a final year aerospace enineering student at the Airforce Institute of Technology Kaduna State.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <div class="info-card-title">▸ Our Strategic Contribution</div>
        <div class="info-card-value">
            <ul class="bullet-list">
                <li>Enhanced aircraft maintenance and platform readiness</li>
                <li>Reduced dependence on foreign diagnostic systems</li>
                <li>Stronger national defense capabilities</li>
                <li>Low-cost indigenous aerospace technology development</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    try:
        img = Image.open("images/Startup company name.png")
        st.markdown('<div class="img-frame">', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    except:
        st.markdown("""
        <div style="background:#0a1628; border:1px solid #0ea5e920; border-radius:12px;
        padding:5rem 2rem; text-align:center; color:#0ea5e9;">
            <div style="font-size:4rem;">🛡️</div>
            <div style="font-family:'Orbitron',monospace; font-size:0.75rem;
            letter-spacing:2px; margin-top:1rem;">AIS Technologies</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── VISION & MISSION ──────────────────────────────────────────────────────────
st.markdown('<div class="section-wrapper dark">', unsafe_allow_html=True)

col_v, col_m = st.columns(2, gap="large")

with col_v:
    try:
        img = Image.open("images/fighter jet pic.jpg")
        st.markdown('<div class="img-frame">', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    except:
        st.markdown("""
        <div style="background:#0a1628; border:1px solid #0ea5e920; border-radius:12px;
        padding:4rem 2rem; text-align:center; color:#0ea5e9;">
            <div style="font-size:4rem;">✈️</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-badge">// OUR VISION</div>
    <div class="section-title">Global <span>Impact</span></div>
    <div class="info-card">
        <div class="info-card-value">
            To become a globally respected aerospace and defense systems company delivering
            intelligent technologies that enhance airspace safety, mission effectiveness,
            and national security, while enabling Nigeria as an emerging aerospace nation
            to build and control its own critical capabilities.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_m:
    st.markdown("""
    <div class="section-badge">// OUR MISSION</div>
    <div class="section-title">Engineering <span>Excellence</span></div>
    <div class="info-card">
        <div class="info-card-title">▸ What We Build</div>
        <div class="info-card-value">
            <ul class="bullet-list">
                <li>Aerospace engineering systems</li>
                <li>Intelligent sensing platforms</li>
                <li>Data-driven decision systems</li>
                <li>Artificial intelligence autonomous technologies</li>
                <li>Adaptive aerospace solutions</li>
            </ul>
        </div>
    </div>
    <div class="info-card" style="margin-top:1rem;">
        <div class="info-card-title">▸ Designed For</div>
        <div class="info-card-value">
            Civil aviation &nbsp;•&nbsp; National defense &nbsp;•&nbsp;
            Strategic aerospace applications &nbsp;•&nbsp;
            MRO operators &nbsp;•&nbsp; Defense establishments
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── ASHMIS PLATFORM ───────────────────────────────────────────────────────────
st.markdown('<div class="section-wrapper mid">', unsafe_allow_html=True)

st.markdown("""
<div class="section-badge">// FIRST PLATFORM</div>
<div class="section-title">Aircraft Structural Health Monitoring<br>
<span>& Integrated System (ASHMIS)</span></div>
<div class="section-desc">
    Our inaugural platform transforming aircraft maintenance and inspection
    through thermal-visual imaging fusion and computer vision AI.
</div>
""", unsafe_allow_html=True)

col_img, col_text = st.columns([2, 3], gap="large")

with col_img:
    try:
        img = Image.open("images/ASHMIS img.png")
        st.markdown('<div class="img-frame">', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    except:
        st.markdown("""
        <div style="background:#0a1628; border:1px solid #0ea5e920; border-radius:12px;
        padding:4rem 2rem; text-align:center; color:#0ea5e9;">
            <div style="font-size:4rem;">🔍</div>
            <div style="font-family:'Orbitron',monospace; font-size:0.75rem;
            letter-spacing:2px; margin-top:1rem;">ASHMIS</div>
        </div>
        """, unsafe_allow_html=True)

with col_text:
    st.markdown("""
    <div class="info-card">
        <div class="info-card-title">▸ Detection Capability</div>
        <div class="info-card-value">
            <ul class="bullet-list">
                <li>Structural cracks and micro-fractures</li>
                <li>Delamination in composite structures</li>
                <li>Corrosion and oxidation patterns</li>
                <li>Material fatigue and hidden defects</li>
            </ul>
        </div>
    </div>
    <div class="info-card">
        <div class="info-card-title">▸ Operational Impact</div>
        <div class="info-card-value">
            <ul class="bullet-list">
                <li>Faster, AI-driven inspections</li>
                <li>Early fault detection before failure</li>
                <li>Reduced Aircraft on Ground (AOG) time</li>
                <li>Improved airworthiness decision-making</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── WHY IT MATTERS ────────────────────────────────────────────────────────────
st.markdown('<div class="section-wrapper dark">', unsafe_allow_html=True)

st.markdown("""
<div class="section-badge">// THE PROBLEM WE SOLVE</div>
<div class="section-title">Why It <span>Matters</span></div>
<div class="section-desc">
    In many regions especially across Africa aircraft maintenance still relies heavily
    on outdated methods. VELTHORIS Technologies is changing that reality.
</div>
""", unsafe_allow_html=True)

col_prob, col_sol = st.columns(2, gap="large")

with col_prob:
    st.markdown("""
    <div class="info-card">
        <div class="info-card-title">▸ Current Industry Challenges</div>
        <div class="info-card-value">
            <ul class="bullet-list">
                <li>Manual inspections prone to human error</li>
                <li>Time-based maintenance schedules</li>
                <li>Limited access to advanced diagnostic tools</li>
                <li>High foreign dependency for MRO capabilities</li>
                <li>High Aircraft on Ground (AOG) costs</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_sol:
    st.markdown("""
    <div class="info-card">
        <div class="info-card-title">▸ What VELTHORIS Technologies Delivers</div>
        <div class="info-card-value">
            <ul class="bullet-list">
                <li>Increased aviation safety through AI detection</li>
                <li>Reduced operational and maintenance costs</li>
                <li>Stronger local aerospace capability</li>
                <li>Support for airlines, MROs, and regulators</li>
                <li>Safer skies and more self-reliant aerospace ecosystems</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

try:
    img = Image.open("images/global-air-travel-connectivity.jpg")
    st.markdown('<div class="img-frame" style="margin-top:2rem;">', unsafe_allow_html=True)
    st.image(img, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
except:
    pass

st.markdown('</div>', unsafe_allow_html=True)

# ── ROADMAP ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-wrapper mid">', unsafe_allow_html=True)

st.markdown("""
<div class="section-badge">// BEYOND MAINTENANCE</div>
<div class="section-title">Long-Term <span>Roadmap</span></div>
<div class="section-desc">
    ASHMIS is our first platform not our destination. VELTHORIS Technologies is building
    a systems-level aerospace and defense technology portfolio.
</div>
<div class="roadmap-grid">
    <div class="roadmap-item">
        <div class="roadmap-num">01 // Now</div>
        <div class="roadmap-title">ASHMIS</div>
        <div class="roadmap-desc">AI-powered structural health monitoring and NDT inspection platform for civil and military aviation.</div>
    </div>
    <div class="roadmap-item">
        <div class="roadmap-num">02 // Near-Term</div>
        <div class="roadmap-title">Intelligent Aircraft Systems</div>
        <div class="roadmap-desc">Embedded smart sensor networks and real-time airframe condition monitoring across aircraft types.</div>
    </div>
    <div class="roadmap-item">
        <div class="roadmap-num">03 // Mid-Term</div>
        <div class="roadmap-title">Aerospace Data Platforms</div>
        <div class="roadmap-desc">Fleet-wide analytics, predictive maintenance dashboards, and MRO decision intelligence systems.</div>
    </div>
    <div class="roadmap-item">
        <div class="roadmap-num">04 // Long-Term</div>
        <div class="roadmap-title">Defense & Surveillance</div>
        <div class="roadmap-desc">Advanced ISR systems, defense-grade AI, autonomous inspection drones, and national security technologies.</div>
    </div>
</div>
""", unsafe_allow_html=True)

try:
    img = Image.open("images/airplane-night-runway.jpg")
    st.markdown('<div class="img-frame" style="margin-top:3rem;">', unsafe_allow_html=True)
    st.image(img, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
except:
    pass

st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div>
        <div class="footer-brand">VELTHORIS Technologies</div>
        <div class="footer-info">Aerospace Intelligent Systems · AFIT Kaduna · Built for Impact.</div>
               <div class="footer-info"> Founded by Ilim A.Theophilus</div>
    </div>
    <div style="font-family:'Orbitron',monospace; font-size:0.65rem;
    color:#334155; letter-spacing:2px; text-transform:uppercase;">
        ASHMIS v1.0 · Powered by DRONE VISION-AI
    </div>
</div>
""", unsafe_allow_html=True)
