import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="ASHMIS Platform",
    layout="wide",
    page_icon="✈️"
)

# ── GLOBAL DESIGN SYSTEM ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg: #0B1220;
    --surface: #111827;
    --glass: rgba(255,255,255,0.04);
    --border: rgba(255,255,255,0.08);

    --primary: #3B82F6;
    --accent: #06B6D4;
    --highlight: #8B5CF6;

    --text-main: #E5E7EB;
    --text-sub: #9CA3AF;
}

.stApp {
    background: linear-gradient(135deg, #0B1220, #0F172A);
    color: var(--text-main);
    font-family: 'Inter', sans-serif;
}

* { transition: all 0.3s ease; }

/* Cards */
.card {
    background: var(--glass);
    border: 1px solid var(--border);
    backdrop-filter: blur(14px);
    border-radius: 16px;
    padding: 1.5rem;
}

/* Title */
.title {
    font-family: 'Orbitron', monospace;
    font-weight: 700;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--primary), var(--accent));
    border: none;
    border-radius: 10px;
    color: white;
    font-weight: 600;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(59,130,246,0.3);
}

/* Feature cards */
.feature {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem;
}

.feature:hover {
    transform: translateY(-5px);
}

/* Footer */
.footer {
    text-align:center;
    color: var(--text-sub);
    margin-top:2rem;
    font-size:0.8rem;
}

/* Hide streamlit */
#MainMenu, footer, header {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── HERO ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="card">
    <div class="title" style="font-size:2.5rem;">
        ASHMIS <span style="color:#06B6D4;">Platform</span>
    </div>
    <div style="color:#9CA3AF; margin-top:0.5rem; max-width:600px;">
        Aircraft Structural Health Monitoring system integrating computer vision,
        thermal imaging, and predictive analytics for defect detection.
    </div>
</div>
""", unsafe_allow_html=True)

# ── FEATURES ─────────────────────────────────────────────────────────────
st.markdown("### System Capabilities")

col1, col2, col3 = st.columns(3)

def feature(title, desc):
    st.markdown(f"""
    <div class="feature">
        <b>{title}</b>
        <p style="color:#9CA3AF;">{desc}</p>
    </div>
    """, unsafe_allow_html=True)

with col1:
    feature("Defect Detection", "Detect cracks, corrosion, delamination using CV pipeline")

with col2:
    feature("Thermal + Visual Fusion", "Combine infrared and RGB imaging")

with col3:
    feature("Inspection Reports", "Generate aviation-grade inspection outputs")

# ── IMAGE PREVIEW ────────────────────────────────────────────────────────
st.markdown("### System Preview")

try:
    img = Image.open("images/ASHMIS img.png")
    st.image(img, use_container_width=True)
except:
    st.info("Add preview image to /images folder")

# ── FOOTER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
ASHMIS — Aircraft Structural Health Monitoring System | AFIT Kaduna
</div>
""", unsafe_allow_html=True)
