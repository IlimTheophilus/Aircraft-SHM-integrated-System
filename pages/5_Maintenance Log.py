import streamlit as st
import pandas as pd
from datetime import datetime, date
import os
import hashlib
import io

st.set_page_config(
    page_title="ASHMIS — Maintenance Log",
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
    letter-spacing: 3px; text-transform: uppercase; margin-bottom: 1.2rem;
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
    letter-spacing: 3px; text-transform: uppercase; margin-bottom: 0.8rem;
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
    max-width: 700px; line-height: 1.8; margin-bottom: 2rem;
}

/* ── Form Panel ── */
.form-panel {
    background: #0a1628;
    border: 1px solid #0ea5e920;
    border-radius: 12px; padding: 2rem;
    position: relative; overflow: hidden; margin-bottom: 2rem;
}
.form-panel::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #0ea5e9, transparent);
}
.form-panel-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem; color: #38bdf8;
    letter-spacing: 3px; text-transform: uppercase; margin-bottom: 1.5rem;
    display: flex; align-items: center; gap: 0.6rem;
}

/* ── Pulse Dot ── */
.pulse-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #10b981; display: inline-block;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(16,185,129,0.4); }
    50%       { opacity: 0.7; box-shadow: 0 0 0 6px rgba(16,185,129,0); }
}

/* ── Logs Panel ── */
.logs-panel {
    background: #0a1628;
    border: 1px solid #0ea5e920;
    border-radius: 12px; padding: 2rem;
    position: relative; overflow: hidden; margin-top: 2rem;
}
.logs-panel::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #10b981, transparent);
}
.logs-panel-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem; color: #10b981;
    letter-spacing: 3px; text-transform: uppercase; margin-bottom: 1.5rem;
    display: flex; align-items: center; gap: 0.6rem;
}
.log-row-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem; color: #94a3b8;
    padding: 0.5rem 0; border-bottom: 1px solid #0ea5e910;
}

/* ── Streamlit element overrides ── */
.stTextInput input, .stNumberInput input, .stTextArea textarea,
.stSelectbox select, .stDateInput input {
    background: #050f1f !important; border: 1px solid #0ea5e920 !important;
    color: #e2e8f0 !important; font-family: 'Inter', sans-serif !important;
    border-radius: 6px !important; font-size: 0.88rem !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #0ea5e9 !important; box-shadow: 0 0 0 2px rgba(14,165,233,0.15) !important;
}
.stTextInput label, .stNumberInput label, .stTextArea label,
.stDateInput label, .stSelectbox label {
    font-family: 'Inter', sans-serif !important; font-size: 0.78rem !important;
    color: #64748b !important; letter-spacing: 0.5px !important;
    text-transform: uppercase !important; font-weight: 600 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #0369a1, #0ea5e9) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important; font-weight: 700 !important;
    letter-spacing: 1px !important; text-transform: uppercase !important;
    font-size: 0.82rem !important; box-shadow: 0 0 20px rgba(14,165,233,0.2) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    box-shadow: 0 0 35px rgba(14,165,233,0.4) !important; transform: translateY(-1px) !important;
}
button[kind="secondary"] {
    background: rgba(239, 68, 68, 0.1) !important;
    border: 1px solid rgba(239,68,68,0.3) !important;
    color: #f87171 !important; box-shadow: none !important;
}
.stDataFrame {
    background: #050f1f !important; border: 1px solid #0ea5e915 !important; border-radius: 8px !important;
}

/* ── Compliance bar ── */
.compliance-bar {
    background: rgba(14,165,233,0.05);
    border: 1px solid rgba(14,165,233,0.15);
    border-radius: 10px; padding: 1rem 1.5rem;
    display: flex; align-items: center;
    gap: 1.5rem; flex-wrap: wrap; margin-bottom: 2rem;
}
.compliance-item {
    display: flex; align-items: center; gap: 0.4rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem; color: #64748b;
    letter-spacing: 1px; text-transform: uppercase; font-weight: 600;
}
.compliance-dot { color: #0ea5e9; }

/* ── Sidebar info ── */
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
    <div class="hero-scan-line"></div>
    <div class="hero-badge">⬡ &nbsp; Aviation-Standard Maintenance Record System &nbsp; ⬡</div>
    <div class="hero-title">
        Predictive Maintenance<br>
        <span>Log System</span>
    </div>
    <div class="hero-subtitle">ASHMIS — MRO Airworthiness Record Module</div>
    <div class="hero-desc">
        Record, track, and manage aircraft maintenance events following standard aviation
        log practices. Capture defects, corrective actions, technician certification,
        and schedule next inspection — per NCAA, EASA, and ICAO standards.
    </div>
    <div class="hero-tag">
        <span class="hero-tag-item"><span class="hero-tag-dot"></span> NCAA Nig.CARs</span>
        <span class="hero-tag-item"><span class="hero-tag-dot"></span> EASA Part-145</span>
        <span class="hero-tag-item"><span class="hero-tag-dot"></span> ICAO Annex 6</span>
        <span class="hero-tag-item"><span class="hero-tag-dot"></span> MIL-STD-1629</span>
        <span class="hero-tag-item"><span class="hero-tag-dot"></span> Duplicate Detection</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem; font-family:'Orbitron',monospace;
    font-size:0.65rem; color:#38bdf8; letter-spacing:3px; text-transform:uppercase;">
        // Log System Info
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        <div class="info-card-title">▸ Standards Compliant</div>
        <div class="info-card-value">NCAA Nig.CARs · EASA Part-M<br>EASA Part-145 · ICAO Annex 6<br>MIL-STD-1629 · MSG-3</div>
    </div>
    <div class="info-card">
        <div class="info-card-title">▸ Features</div>
        <div class="info-card-value">
            Duplicate Hash Detection<br>
            Per-Row Edit / Delete<br>
            CSV Export per Entry<br>
            Scheduled Maintenance Tracking<br>
            Technician License Capture
        </div>
    </div>
    <div class="info-card">
        <div class="info-card-title">▸ Storage</div>
        <div class="info-card-value">Local CSV · data/maintenance_logs.csv</div>
    </div>
    """, unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
for key in ["show_logs", "pending_entry", "duplicate_detected",
            "row_editing", "row_options_open", "confirm_delete", "confirm_clear_all"]:
    if key not in st.session_state:
        st.session_state[key] = None if key in [
            "pending_entry", "row_editing", "row_options_open", "confirm_delete"] else False

# ── HELPERS ───────────────────────────────────────────────────────────────────
def generate_entry_hash(entry: dict) -> str:
    unique_string = (
        str(entry.get("Aircraft Registration", "")) +
        str(entry.get("Maintenance Date", "")) +
        str(entry.get("Maintenance Time", "")) +
        str(entry.get("Defect Detected", "")) +
        str(entry.get("Corrective Action", ""))
    )
    return hashlib.sha256(unique_string.encode()).hexdigest()

def load_logs():
    csv_path = "data/maintenance_logs.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()

def save_logs(df):
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/maintenance_logs.csv", index=False)

def reset_duplicate_state():
    st.session_state.duplicate_detected = False
    st.session_state.pending_entry = None

# ── FORM SECTION ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-wrapper mid">', unsafe_allow_html=True)

st.markdown("""
<div class="section-badge">// NEW MAINTENANCE EVENT</div>
<div class="section-title">Log <span>Maintenance Entry</span></div>
<div class="section-desc">
    Complete all mandatory fields to create a new maintenance record.
    Aircraft Registration, Defect Description, and Corrective Action are required.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="compliance-bar">
    <div class="compliance-item"><span class="compliance-dot">●</span> NCAA Nig.CARs</div>
    <div class="compliance-item"><span class="compliance-dot">●</span> EASA Part-M</div>
    <div class="compliance-item"><span class="compliance-dot">●</span> EASA Part-145</div>
    <div class="compliance-item"><span class="compliance-dot">●</span> ICAO Annex 6</div>
    <div class="compliance-item"><span class="compliance-dot">●</span> MIL-STD-1629</div>
    <div class="compliance-item"><span class="compliance-dot">●</span> MSG-3</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="form-panel">', unsafe_allow_html=True)
st.markdown("""
<div class="form-panel-title">
    <span class="pulse-dot"></span> // Maintenance Record Entry Form
</div>
""", unsafe_allow_html=True)

with st.form("maintenance_form"):
    col_a, col_b, col_c = st.columns(3, gap="medium")

    with col_a:
        aircraft_reg = st.text_input("Aircraft Registration (e.g., 5N-ABC)", max_chars=10)
        aircraft_type = st.text_input("Aircraft Type (e.g., Cessna 172)")
        flight_hours = st.number_input("Flight Hours / Time in Service", min_value=0.0, step=0.1)
        tech_name = st.text_input("Technician Name")
        tech_license = st.text_input("Technician License / ID")

    with col_b:
        date_performed = st.date_input("Maintenance Date", datetime.today())
        time_performed = st.text_input("Time Performed (e.g., 02:30 PM)",
                                       datetime.now().strftime("%I:%M %p"))
        next_sched = st.date_input("Next Scheduled Maintenance (optional)",
                                   None, min_value=date.today())
        parts_replaced = st.text_input("Parts Replaced (if any)")

    with col_c:
        defect_detected = st.text_area("Defect Detected / Issue Description", height=130)
        corrective_action = st.text_area("Corrective Action Taken", height=130)

    submitted = st.form_submit_button("💾 Save Maintenance Log", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

if submitted:
    if not aircraft_reg or not defect_detected or not corrective_action:
        st.warning("⚠️ Please fill Aircraft Registration, Defect Detected, and Corrective Action.")
    else:
        entry = {
            "Aircraft Registration": aircraft_reg,
            "Aircraft Type": aircraft_type,
            "Flight Hours": flight_hours,
            "Maintenance Date": date_performed.strftime("%Y-%m-%d"),
            "Maintenance Time": time_performed,
            "Defect Detected": defect_detected,
            "Corrective Action": corrective_action,
            "Parts Replaced": parts_replaced,
            "Next Scheduled Maintenance": next_sched.strftime("%Y-%m-%d") if next_sched else "",
            "Technician Name": tech_name,
            "Technician License/ID": tech_license,
            "Record Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        entry["Entry Hash"] = generate_entry_hash(entry)
        df = load_logs()

        if "Entry Hash" in df.columns and entry["Entry Hash"] in df["Entry Hash"].values:
            st.session_state.pending_entry = entry
            st.session_state.duplicate_detected = True
        else:
            df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
            save_logs(df)
            st.success("✅ Maintenance log saved successfully.")

if st.session_state.duplicate_detected:
    st.warning("⚠️ A duplicate maintenance log entry was detected.")
    st.json(st.session_state.pending_entry)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Save Anyway"):
            df = load_logs()
            df = pd.concat([df, pd.DataFrame([st.session_state.pending_entry])], ignore_index=True)
            save_logs(df)
            st.success("✅ Entry saved.")
            reset_duplicate_state()
    with col2:
        if st.button("❌ Cancel"):
            st.info("Duplicate entry discarded.")
            reset_duplicate_state()

st.markdown('</div>', unsafe_allow_html=True)

# ── SAVED LOGS SECTION ────────────────────────────────────────────────────────
st.markdown('<div class="section-wrapper dark">', unsafe_allow_html=True)

st.markdown("""
<div class="section-badge">// AIRWORTHINESS RECORDS</div>
<div class="section-title">Maintenance <span>Log Database</span></div>
<div class="section-desc">
    View, edit, export, and manage all saved maintenance events.
    Each record is hash-verified for data integrity.
</div>
""", unsafe_allow_html=True)

if st.button("📂 Toggle Saved Logs", use_container_width=False):
    st.session_state.show_logs = not st.session_state.show_logs

if st.session_state.show_logs:
    df_logs = load_logs()

    if not df_logs.empty:
        st.markdown('<div class="logs-panel">', unsafe_allow_html=True)
        st.markdown("""
        <div class="logs-panel-title">
            <span class="pulse-dot" style="background:#10b981;"></span>
            // Maintenance Log Records
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(df_logs, use_container_width=True)

        st.markdown("""
        <div style="font-family:'Orbitron',monospace; font-size:0.62rem;
        color:#38bdf8; letter-spacing:2px; text-transform:uppercase;
        margin: 1.5rem 0 1rem;">// Row Actions</div>
        """, unsafe_allow_html=True)

        for idx, row in df_logs.iterrows():
            row_cols = st.columns([0.80, 0.20])
            with row_cols[0]:
                st.markdown(
                    f'<div class="log-row-label">Row {idx}: '
                    f'<strong style="color:#38bdf8;">{row["Aircraft Registration"]}</strong>'
                    f' — {row["Maintenance Date"]} — {row.get("Defect Detected","")[:60]}…</div>',
                    unsafe_allow_html=True
                )
            with row_cols[1]:
                if st.button("⚙️ Options", key=f"options_{idx}"):
                    st.session_state.row_options_open = idx

            if st.session_state.row_options_open == idx:
                opt_cols = st.columns(3)
                with opt_cols[0]:
                    if st.button("✏️ Edit", key=f"edit_{idx}"):
                        st.session_state.row_editing = idx
                        st.session_state.row_options_open = None
                with opt_cols[1]:
                    csv_buf = io.StringIO()
                    df_logs.loc[[idx]].to_csv(csv_buf, index=False)
                    st.download_button(
                        "⬇️ Export",
                        csv_buf.getvalue(),
                        f"{row['Aircraft Registration']}_{row['Maintenance Date']}.csv",
                        "text/csv",
                        key=f"dl_{idx}"
                    )
                with opt_cols[2]:
                    if st.button("🗑 Delete", key=f"delete_{idx}"):
                        st.session_state.confirm_delete = idx
                        st.session_state.row_options_open = None

        if st.session_state.row_editing is not None:
            edit_idx = st.session_state.row_editing
            edit_row = df_logs.loc[edit_idx].copy()

            st.markdown(f"""
            <div style="font-family:'Orbitron',monospace; font-size:0.65rem;
            color:#f59e0b; letter-spacing:2px; text-transform:uppercase;
            margin: 2rem 0 1rem;">// Editing Row {edit_idx} — {edit_row['Aircraft Registration']}</div>
            """, unsafe_allow_html=True)

            edit_cols = st.columns(4)
            with edit_cols[0]:
                edit_row["Aircraft Registration"] = st.text_input(
                    "Registration", value=edit_row["Aircraft Registration"])
                edit_row["Aircraft Type"] = st.text_input(
                    "Type", value=edit_row["Aircraft Type"])
                edit_row["Flight Hours"] = st.number_input(
                    "Flight Hours", min_value=0.0, step=0.1,
                    value=float(edit_row["Flight Hours"]))
            with edit_cols[1]:
                edit_row["Maintenance Date"] = st.date_input(
                    "Maintenance Date",
                    value=datetime.strptime(edit_row["Maintenance Date"], "%Y-%m-%d").date())
                edit_row["Maintenance Time"] = st.text_input(
                    "Time Performed", value=edit_row["Maintenance Time"])
                edit_row["Next Scheduled Maintenance"] = st.date_input(
                    "Next Scheduled Maintenance",
                    value=datetime.strptime(
                        edit_row["Next Scheduled Maintenance"], "%Y-%m-%d").date()
                    if edit_row["Next Scheduled Maintenance"] else None,
                    min_value=date.today())
            with edit_cols[2]:
                edit_row["Defect Detected"] = st.text_area(
                    "Defect Detected", value=edit_row["Defect Detected"])
                edit_row["Corrective Action"] = st.text_area(
                    "Corrective Action", value=edit_row["Corrective Action"])
            with edit_cols[3]:
                edit_row["Parts Replaced"] = st.text_input(
                    "Parts Replaced", value=edit_row["Parts Replaced"])
                edit_row["Technician Name"] = st.text_input(
                    "Technician Name", value=edit_row["Technician Name"])
                edit_row["Technician License/ID"] = st.text_input(
                    "Technician License/ID", value=edit_row["Technician License/ID"])

            if st.button("💾 Save Row Changes"):
                edit_row["Maintenance Date"] = edit_row["Maintenance Date"].strftime("%Y-%m-%d")
                edit_row["Next Scheduled Maintenance"] = (
                    edit_row["Next Scheduled Maintenance"].strftime("%Y-%m-%d")
                    if edit_row["Next Scheduled Maintenance"] else ""
                )
                df_logs.loc[edit_idx] = edit_row
                df_logs["Entry Hash"] = df_logs.apply(lambda x: generate_entry_hash(x), axis=1)
                save_logs(df_logs)
                st.success("✅ Row updated successfully.")
                st.session_state.row_editing = None
                st.rerun()

        if st.session_state.confirm_delete is not None:
            st.warning(f"⚠️ Confirm deletion of Row {st.session_state.confirm_delete}?")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Yes, Delete", key="confirm_delete_yes"):
                    df_logs = df_logs.drop(st.session_state.confirm_delete)
                    save_logs(df_logs)
                    st.success("✅ Entry deleted.")
                    st.session_state.confirm_delete = None
                    st.rerun()
            with c2:
                if st.button("❌ No, Keep", key="confirm_delete_no"):
                    st.session_state.confirm_delete = None

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="margin-top:1.5rem;">', unsafe_allow_html=True)
        if st.button("🗑 Clear All Log Entries"):
            st.session_state.confirm_clear_all = True

        if st.session_state.confirm_clear_all:
            st.warning("⚠️ This will permanently erase all maintenance records. Proceed?")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Clear All Records"):
                    os.remove("data/maintenance_logs.csv")
                    st.success("✅ All records cleared.")
                    st.session_state.confirm_clear_all = False
                    st.rerun()
            with c2:
                if st.button("❌ Cancel"):
                    st.session_state.confirm_clear_all = False
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:#0a1628; border:1px solid #0ea5e920; border-radius:10px;
        padding:3rem; text-align:center; color:#475569; font-family:'Inter',sans-serif;
        font-size:0.9rem; letter-spacing:1px;">
            No maintenance logs saved yet. Submit the form above to create the first entry.
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
