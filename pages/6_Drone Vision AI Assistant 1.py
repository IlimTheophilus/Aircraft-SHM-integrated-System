import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="SHM AI Assistant", layout="wide")
st.title("🤖 Aircraft SHM AI Assistant")

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

# Configure Gemini
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel(
    model_name="model_name="gemini-2.0-flash",",
    system_instruction=SYSTEM_PROMPT
)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.gemini_chat = model.start_chat(history=[])

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_prompt = st.chat_input("Ask about aircraft SHM, defects, sensors, or maintenance...")

if user_prompt:
    # Show user message
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Get Gemini response
    with st.chat_message("assistant"):
        with st.spinner("Analysing..."):
            response = st.session_state.gemini_chat.send_message(user_prompt)
            reply = response.text
            st.markdown(reply)

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
