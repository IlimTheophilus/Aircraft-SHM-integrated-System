# Pages/7_Drone Vision AI Assistant 2.py

import streamlit as st
from openai import OpenAI, RateLimitError
import requests
import os
import time

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(page_title="Drone Vision AI Assistant", layout="wide")
st.title("🤖 Aircraft SHM AI Assistant")

# -------------------------------
# API keys
# -------------------------------
# Make sure to set these in Streamlit Cloud Secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
HF_API_TOKEN = st.secrets["HF_API_KEY"]

client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------------
# Chat history
# -------------------------------
if "ai_messages" not in st.session_state:
    st.session_state.ai_messages = [
        {"role": "system", "content": "You are a helpful AI assistant specialized in aircraft SHM and maintenance."}
    ]

MAX_MESSAGES = 8  # limit chat history
MAX_TOKENS = 300  # limit response length

# -------------------------------
# Function to call OpenAI
# -------------------------------
def get_ai_response(messages):
    # Keep history within limits
    if len(messages) > MAX_MESSAGES + 1:
        messages = [messages[0]] + messages[-MAX_MESSAGES:]

    try:
        time.sleep(1)  # throttle repeated calls
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=MAX_TOKENS
        )
        return response.choices[0].message.content

    except RateLimitError:
        return "⚠️ The AI service is busy. Please wait a few seconds and try again."

# -------------------------------
# Function for Hugging Face TTS
# -------------------------------
def get_hf_tts_audio(text, filename="output.wav"):
    url = "https://api-inference.huggingface.co/models/facebook/fastspeech2-en-ljspeech"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": text}

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        audio_bytes = response.content
        with open(filename, "wb") as f:
            f.write(audio_bytes)
        return filename
    else:
        st.warning("⚠️ Hugging Face TTS API error")
        return None

# -------------------------------
# User input
# -------------------------------
user_input = st.text_input("Type your question about Aircraft SHM:")

if st.button("Send") and user_input.strip():
    st.session_state.ai_messages.append({"role": "user", "content": user_input})
    ai_response = get_ai_response(st.session_state.ai_messages)
    st.session_state.ai_messages.append({"role": "assistant", "content": ai_response})
    
    # Display chat
    for msg in st.session_state.ai_messages[1:]:  # skip system message
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**AI:** {msg['content']}")

    # Play TTS
    audio_file = get_hf_tts_audio(ai_response)
    if audio_file:
        st.audio(audio_file, format="audio/wav")

# -------------------------------
# Show chat history sidebar
# -------------------------------
with st.sidebar:
    st.header("Chat History")
    for msg in st.session_state.ai_messages[1:]:
        role = "You" if msg["role"] == "user" else "AI"
        st.write(f"{role}: {msg['content'][:100]}{'...' if len(msg['content'])>100 else ''}")

