# pages/Drone_Vision_HF_Assistant.py

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pathlib import Path
import time
import base64

# ---------------------------
# CONFIG
# ---------------------------
HF_MODEL = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"  # Change to your chosen HF model
HF_TTS_MODEL = "facebook/fastspeech2-en-ljspeech"
MAX_MESSAGES = 8
MAX_TOKENS = 300
THROTTLE = 1  # seconds between requests

# ---------------------------
# STREAMLIT SETUP
# ---------------------------
st.set_page_config(page_title="Aircraft SHM AI Assistant", layout="wide")
st.title("🤖 Aircraft SHM AI Assistant (Hugging Face)")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "system", "content": "You are an expert in Aircraft Structural Health Monitoring (SHM)."}]

# ---------------------------
# HUGGING FACE AUTH
# ---------------------------
HF_API_TOKEN = st.secrets.get("HF_API_KEY", None)
if HF_API_TOKEN is None:
    st.error("⚠️ HF_API_KEY missing. Add it to Streamlit secrets.")
    st.stop()

# ---------------------------
# LOAD MODELS
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    # Chat model
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, use_auth_token=HF_API_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL, use_auth_token=HF_API_TOKEN)
    # TTS pipeline
    tts_pipeline = pipeline("text-to-speech", model=HF_TTS_MODEL, use_auth_token=HF_API_TOKEN)
    return tokenizer, model, tts_pipeline

tokenizer, model, tts_pipeline = load_models()

# ---------------------------
# FUNCTIONS
# ---------------------------
def generate_hf_response(user_input):
    # Append user input
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Limit chat history
    messages = st.session_state.chat_history
    if len(messages) > MAX_MESSAGES + 1:
        messages = [messages[0]] + messages[-MAX_MESSAGES:]
    
    # Prepare prompt for HF model
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt += f"{content}\n"
        elif role == "user":
            prompt += f"User: {content}\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n"
    
    try:
        time.sleep(THROTTLE)
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=MAX_TOKENS)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Append assistant response to history
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
        return response_text
    except Exception as e:
        return f"⚠️ Hugging Face AI error: {e}"

def text_to_speech(audio_text):
    try:
        audio_output = tts_pipeline(audio_text)
        audio_bytes = audio_output["audio"]
        b64_audio = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
            <audio autoplay="true" controls>
                <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav">
            </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"⚠️ Hugging Face TTS error: {e}")

# ---------------------------
# USER INPUT
# ---------------------------
user_input = st.text_input("Type your question about Aircraft SHM:")

if st.button("Send") and user_input:
    # Generate AI response
    ai_response = generate_hf_response(user_input)
    st.markdown(f"**You:** {user_input}")
    st.markdown(f"**AI:** {ai_response}")
    
    # Play TTS
    text_to_speech(ai_response)

# ---------------------------
# SHOW CHAT HISTORY
# ---------------------------
st.markdown("---")
st.subheader("Chat History")
for msg in st.session_state.chat_history[1:]:  # skip system prompt
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**AI:** {msg['content']}")
