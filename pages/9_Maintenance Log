import streamlit as st
import pandas as pd
from datetime import datetime, date
import os
import hashlib
import io

st.set_page_config(page_title="Maintenance Login", layout="centered")
st.title("✈️ Predictive / Maintenance Login Form")
st.write(
    """
    Use this form to enter maintenance events and defects following standard
    aviation maintenance log practices.
    """
)

# ---------------- Session State ----------------
for key in ["show_logs", "pending_entry", "duplicate_detected",
            "row_editing", "row_options_open", "confirm_delete", "confirm_clear_all"]:
    if key not in st.session_state:
        st.session_state[key] = None if key in [
            "pending_entry", "row_editing", "row_options_open", "confirm_delete"] else False

# ---------------- Helper Functions ----------------


def generate_entry_hash(entry: dict) -> str:
    """Creates a unique fingerprint for a maintenance entry"""
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


# ---------------- Form Inputs for New Entry ----------------
with st.form("maintenance_form"):
    aircraft_reg = st.text_input(
        "📌 Aircraft Registration (e.g., 5N-ABC)", max_chars=10)
    aircraft_type = st.text_input("📌 Aircraft Type (e.g., Cessna 172)")
    flight_hours = st.number_input(
        "🕐 Flight Hours / Time in Service", min_value=0.0, step=0.1)
    date_performed = st.date_input("📅 Maintenance Date", datetime.today())
    time_performed = st.text_input(
        "⏱ Time Performed (e.g., 02:30 PM)", datetime.now().strftime("%I:%M %p"))
    defect_detected = st.text_area("🛠 Defect Detected / Issue Description")
    corrective_action = st.text_area("🔧 Corrective Action Taken")
    parts_replaced = st.text_input("🔩 Parts Replaced (if any)")
    next_sched = st.date_input(
        "📆 Next Scheduled Maintenance (optional)", None, min_value=date.today())
    tech_name = st.text_input("👤 Technician Name")
    tech_license = st.text_input("🪪 Technician License/ID (if applicable)")

    submitted = st.form_submit_button("💾 Save Maintenance Log")

# ---------------- Handle New Entry Submission ----------------
if submitted:
    if not aircraft_reg or not defect_detected or not corrective_action:
        st.warning(
            "Please fill at least Aircraft Registration, Defect, and Action Taken.")
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

        # Check for duplicates
        if "Entry Hash" in df.columns and entry["Entry Hash"] in df["Entry Hash"].values:
            st.session_state.pending_entry = entry
            st.session_state.duplicate_detected = True
        else:
            df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
            save_logs(df)
            st.success("✅ Maintenance log saved successfully.")

# ---------------- Duplicate Warning ----------------
if st.session_state.duplicate_detected:
    st.warning("⚠️ Maintenance log entry already exists.")
    st.json(st.session_state.pending_entry)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Save Again"):
            df = load_logs()
            df = pd.concat(
                [df, pd.DataFrame([st.session_state.pending_entry])], ignore_index=True)
            save_logs(df)
            st.success("✅ Maintenance log saved again.")
            reset_duplicate_state()
    with col2:
        if st.button("❌ Cancel"):
            st.info("Duplicate entry not saved.")
            reset_duplicate_state()

# ---------------- Display Saved Logs ----------------
st.divider()
if st.button("📂 Saved Logs"):
    st.session_state.show_logs = not st.session_state.show_logs

if st.session_state.show_logs:
    df_logs = load_logs()
    if not df_logs.empty:
        # Table remains unchanged
        st.dataframe(df_logs, use_container_width=True)

        st.write("### Row Options (Next to each row)")
        # Loop through each row and align the options button with the row
        for idx, row in df_logs.iterrows():
            row_cols = st.columns([0.85, 0.15])  # Row display + options button
            with row_cols[0]:
                st.write(
                    f"Row {idx}: {row['Aircraft Registration']} - {row['Maintenance Date']}")
            with row_cols[1]:
                if st.button("⚙️", key=f"options_{idx}"):
                    st.session_state.row_options_open = idx

            # Show options under the row when clicked
            if st.session_state.row_options_open == idx:
                option_cols = st.columns(3)
                with option_cols[0]:
                    if st.button("✏️ Edit", key=f"edit_{idx}"):
                        st.session_state.row_editing = idx
                        st.session_state.row_options_open = None
                with option_cols[1]:
                    csv_buf = io.StringIO()
                    df_logs.loc[[idx]].to_csv(csv_buf, index=False)
                    st.download_button("⬇️ Download", csv_buf.getvalue(),
                                       f"{row['Aircraft Registration']}_{row['Maintenance Date']}.csv",
                                       "text/csv")
                with option_cols[2]:
                    if st.button("🗑 Delete", key=f"delete_{idx}"):
                        st.session_state.confirm_delete = idx
                        st.session_state.row_options_open = None

        # ---------------- Row Editing ----------------
        if st.session_state.row_editing is not None:
            edit_idx = st.session_state.row_editing
            edit_row = df_logs.loc[edit_idx].copy()
            st.markdown(
                f"### Editing Row {edit_idx} - {edit_row['Aircraft Registration']}")
            edit_cols = st.columns(4)
            with edit_cols[0]:
                edit_row["Aircraft Registration"] = st.text_input(
                    "Registration", value=edit_row["Aircraft Registration"])
                edit_row["Aircraft Type"] = st.text_input(
                    "Type", value=edit_row["Aircraft Type"])
                edit_row["Flight Hours"] = st.number_input(
                    "Flight Hours", min_value=0.0, step=0.1, value=float(edit_row["Flight Hours"]))
            with edit_cols[1]:
                edit_row["Maintenance Date"] = st.date_input("Maintenance Date",
                                                             value=datetime.strptime(edit_row["Maintenance Date"], "%Y-%m-%d").date())
                edit_row["Maintenance Time"] = st.text_input(
                    "Time Performed", value=edit_row["Maintenance Time"])
                edit_row["Next Scheduled Maintenance"] = st.date_input("Next Scheduled Maintenance",
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
                edit_row["Maintenance Date"] = edit_row["Maintenance Date"].strftime(
                    "%Y-%m-%d")
                if edit_row["Next Scheduled Maintenance"]:
                    edit_row["Next Scheduled Maintenance"] = edit_row["Next Scheduled Maintenance"].strftime(
                        "%Y-%m-%d")
                else:
                    edit_row["Next Scheduled Maintenance"] = ""
                df_logs.loc[edit_idx] = edit_row
                df_logs["Entry Hash"] = df_logs.apply(
                    lambda x: generate_entry_hash(x), axis=1)
                save_logs(df_logs)
                st.success("✅ Row updated successfully.")
                st.session_state.row_editing = None
                st.experimental_rerun()

        # ---------------- Confirm Delete ----------------
        if st.session_state.confirm_delete is not None:
            st.warning("Are you sure you want to delete this entry?")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Yes", key="confirm_delete_yes"):
                    df_logs = df_logs.drop(st.session_state.confirm_delete)
                    save_logs(df_logs)
                    st.success("✅ Entry deleted successfully.")
                    st.session_state.confirm_delete = None
                    st.experimental_rerun()
            with c2:
                if st.button("❌ No", key="confirm_delete_no"):
                    st.session_state.confirm_delete = None

        # ---------------- Clear All Logs ----------------
        if st.button("🗑 Clear All Log Entries"):
            st.session_state.confirm_clear_all = True

        if st.session_state.confirm_clear_all:
            st.warning("Are you sure you want to clear all log entries?")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Clear All"):
                    os.remove("data/maintenance_logs.csv")
                    st.success("✅ All log entries cleared.")
                    st.session_state.confirm_clear_all = False
                    st.experimental_rerun()
            with c2:
                if st.button("❌ Cancel"):
                    st.session_state.confirm_clear_all = False
    else:
        st.info("No maintenance logs saved yet.")
