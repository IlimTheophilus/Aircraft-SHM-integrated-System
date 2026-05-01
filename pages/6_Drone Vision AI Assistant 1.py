import streamlit as st
from groq import Groq

st.set_page_config(page_title="SHM AI Assistant", layout="wide")
st.title("Aircraft SHM AI Assistant")

SYSTEM_PROMPT = """
You are an Aircraft Structural Health Monitoring (SHM) expert assisting aerospace
engineers and technicians. You specialize in:
- Aircraft structures and materials (composites, aluminium alloys, titanium)
- Structural defect types: cracks, corrosion, delamination, fatigue damage
- SHM sensors and signal interpretation (strain gauges, acoustic emission, thermal)
- Non-Destructive Testing (NDT) methods and inspection procedures
- Predictive maintenance and airworthiness regulations
- Nigerian and African aviation context (NCAA, ICAO standards)

Use correct aerospace terminology. Be precise, practical and safety-focused.
If a question is completely unrelated to aircraft SHM or aerospace, politely refuse.
"""

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

for msg in st.session_state.chat_history:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

user_prompt = st.chat_input("Ask about aircraft SHM, defects, sensors, or maintenance...")

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
