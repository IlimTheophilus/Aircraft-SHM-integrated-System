# pages/Drone_Vision_HF_Assistant.py

import streamlit as st
from llama_cpp import Llama
from transformers import pipeline
import time
import base64

# ---------------------------
# CONFIG
# ---------------------------
HF_TTS_MODEL = "facebook/fastspeech2-en-ljspeech"
GGUF_MODEL_PATH = "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"  # download manually or via huggingface_hub
MAX_MESSAGES = 8
MAX_TOKENS = 300
THROTTLE = 1  # seconds

# ---------------------------
# STREAMLIT SETUP
# ---------------------------
st.set_page_config(page_title="Aircraft SHM AI Assistant", layout="wide")
st.title("🤖 Aircraft SHM AI Assistant (Hugging Face GGUF)")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "system", "content": "You are an expert in Aircraft SHM."}]

# ---------------------------
# LOAD MODELS
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    # Llama GGUF model
    llm = Llama(
        model_path=GGUF_MODEL_PATH,
        n_ctx=2048,
        n_threads=8,
        n_gpu_layers=0  # adjust if GPU available
    )
    # Hugging Face TTS
    tts_pipeline = pipeline("text-to-speech", model=HF_TTS_MODEL)
    return llm, tts_pipeline

llm, tts_pipeline = load_models()

# ---------------------------
# FUNCTIONS
# ---------------------------
def generate_hf_response(user_input):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    # Prepare prompt
    messages = st.session_state.chat_history[-MAX_MESSAGES:]
    prompt = ""
    for msg in messages:
        if msg["role"] == "system":
            prompt += f"{msg['content']}\n"
        elif msg["role"] == "user":
            prompt += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            prompt += f"Assistant: {msg['content']}\n"

    time.sleep(THROTTLE)
    try:
        output = llm(prompt, max_tokens=MAX_TOKENS)
        response_text = output["choices"][0]["text"]
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
        return response_text
    except Exception as e:
        return f"⚠️ AI error: {e}"

def text_to_speech(audio_text):
    try:
        audio = tts_pipeline(audio_text)
        audio_bytes = audio["audio"]
        b64_audio = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
        <audio autoplay="true" controls>
            <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav">
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"⚠️ TTS error: {e}")

# ---------------------------
# USER INTERFACE
# ---------------------------
user_input = st.text_input("Ask your question about Aircraft SHM:")

if st.button("Send") and user_input:
    ai_response = generate_hf_response(user_input)
    st.markdown(f"**You:** {user_input}")
    st.markdown(f"**AI:** {ai_response}")
    text_to_speech(ai_response)

# ---------------------------
# SHOW CHAT HISTORY
# ---------------------------
st.markdown("---")
st.subheader("Chat History")
for msg in st.session_state.chat_history[1:]:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**AI:** {msg['content']}")
