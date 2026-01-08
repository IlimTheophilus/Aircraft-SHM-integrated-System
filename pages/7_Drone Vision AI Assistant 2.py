# pages/Drone_Vision_AI_Assistant.py

import streamlit as st
from datetime import datetime
import time
from openai import OpenAI, RateLimitError
from transformers import pipeline

# ===============================
# CONFIG
# ===============================
MAX_MESSAGES = 8       # Chat history limit
MAX_TOKENS = 300       # LLM response token limit
THROTTLE = 1           # Seconds between API calls

# Initialize OpenAI client
openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)

# Initialize Hugging Face TTS
tts = pipeline("text-to-speech", model="facebook/fastspeech2-en-ljspeech")

# ===============================
# SESSION STATE INIT
# ===============================
if "ai_messages" not in st.session_state:
    st.session_state.ai_messages = [
        {"role": "system", "content": "You are an AI assistant specialized in aircraft Structural Health Monitoring (SHM)."}
    ]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ===============================
# FUNCTIONS
# ===============================
def limit_messages(messages):
    """Keep only the first system message + last MAX_MESSAGES user/assistant messages"""
    if len(messages) > MAX_MESSAGES + 1:
        messages = [messages[0]] + messages[-MAX_MESSAGES:]
    return messages

def get_openai_response(messages):
    """Get response from OpenAI GPT"""
    try:
        time.sleep(THROTTLE)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=MAX_TOKENS
        )
        return response.choices[0].message.content
    except RateLimitError:
        return "⚠️ The AI service is busy. Please wait a few seconds and try again."

def get_hf_tts_audio(text, filename="output.wav"):
    """Generate TTS audio using Hugging Face"""
    audio_data = tts(text)
    with open(filename, "wb") as f:
        f.write(audio_data["wav"])
    return filename

def add_message(role, content):
    """Add message to session state"""
    st.session_state.ai_messages.append({"role": role, "content": content})
    st.session_state.ai_messages = limit_messages(st.session_state.ai_messages)
    st.session_state.chat_history.append((role, content))

# ===============================
# STREAMLIT UI
# ===============================
st.title("🤖 Aircraft SHM AI Assistant")

# Display chat history
for role, content in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"**You:** {content}")
    else:
        st.markdown(f"**AI:** {content}")

# User input
user_input = st.text_input("Ask a question about SHM or maintenance:")

if st.button("Send") and user_input:
    # Add user message
    add_message("user", user_input)

    # OpenAI response
    ai_response = get_openai_response(st.session_state.ai_messages)
    add_message("assistant", ai_response)
    
    # Display AI response
    st.markdown(f"**AI:** {ai_response}")

    # Generate voice (cloud-friendly)
    audio_file = get_hf_tts_audio(ai_response)
    st.audio(audio_file, format="audio/wav")
