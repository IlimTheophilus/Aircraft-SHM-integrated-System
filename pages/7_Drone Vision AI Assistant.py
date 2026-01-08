import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="SHM AI Assistant", layout="wide")

st.title("🤖 Aircraft SHM AI Assistant")

SYSTEM_PROMPT = """
You are an Aircraft Structural Health Monitoring (SHM) expert assisting engineers.
You specialize in:
- Aircraft structures and materials
- SHM sensors and signal interpretation
- Damage detection and classification
- Maintenance and inspection recommendations

Use correct aerospace terminology.
If a question is unrelated to aircraft SHM, politely refuse.
"""

client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))

if "ai_messages" not in st.session_state:
    st.session_state.ai_messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

# Display history
for msg in st.session_state.ai_messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

user_prompt = st.chat_input(
    "Ask about aircraft SHM, sensors, or maintenance...")

if user_prompt:
    st.session_state.ai_messages.append(
        {"role": "user", "content": user_prompt}
    )

    with st.chat_message("assistant"):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=st.session_state.ai_messages
        )

        reply = response.choices[0].message.content
        st.markdown(reply)

    st.session_state.ai_messages.append(
        {"role": "assistant", "content": reply}
    )
