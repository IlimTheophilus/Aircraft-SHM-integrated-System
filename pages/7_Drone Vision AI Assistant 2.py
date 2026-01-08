# app.py
import streamlit as st
import requests
import time

# -------------------------------
# CONFIGURATION
# -------------------------------
# replace with your Hugging Face token
HF_API_TOKEN = "YOUR_HUGGINGFACE_API_KEY"
HF_MODEL_URL = "https://api-inference.huggingface.co/models/TheBloke/Mistral-7B-Instruct-v0.2-GGUF"

MAX_MESSAGES = 8      # max chat history to keep
MAX_TOKENS = 300      # max tokens per AI response
THROTTLE_SEC = 1      # wait time between calls

HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# -------------------------------
# STATE INIT
# -------------------------------
if "ai_messages" not in st.session_state:
    st.session_state.ai_messages = []

if "user_messages" not in st.session_state:
    st.session_state.user_messages = []

# -------------------------------
# FUNCTIONS
# -------------------------------


def query_ai(prompt):
    """
    Query Hugging Face hosted Mistral model
    """
    payload = {
        "inputs": f"<s>[INST] {prompt} [/INST]",
        "parameters": {"max_new_tokens": MAX_TOKENS}
    }
    try:
        time.sleep(THROTTLE_SEC)  # throttle API calls
        response = requests.post(HF_MODEL_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        result = response.json()
        # Hugging Face text generation API returns a list with generated_text
        return result[0]["generated_text"]
    except requests.exceptions.HTTPError as e:
        if response.status_code == 429:
            return "⚠️ The AI service is busy. Please wait a few seconds and try again."
        return f"❌ Error: {str(e)}"


def add_to_history(user_msg, ai_msg):
    """
    Store chat history with max limit
    """
    st.session_state.user_messages.append(user_msg)
    st.session_state.ai_messages.append(ai_msg)

    # trim history if too long
    if len(st.session_state.user_messages) > MAX_MESSAGES:
        st.session_state.user_messages = st.session_state.user_messages[-MAX_MESSAGES:]
        st.session_state.ai_messages = st.session_state.ai_messages[-MAX_MESSAGES:]


# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("✈️ Aircraft SHM AI Assistant")

# Display chat history
for user_msg, ai_msg in zip(st.session_state.user_messages, st.session_state.ai_messages):
    st.markdown(f"**You:** {user_msg}")
    st.markdown(f"**AI:** {ai_msg}")

# User input
user_input = st.text_input("Ask your AI assistant about SHM:", "")

if st.button("Send") and user_input.strip():
    ai_response = query_ai(user_input)
    add_to_history(user_input, ai_response)
    st.experimental_rerun()

# -------------------------------
# PLACEHOLDER: Voice input integration
# -------------------------------
st.info("🎤 Voice input coming soon: you can record your question and send it to the AI.")
# Future: integrate with `speech_recognition` or browser-based WebRTC TTS
