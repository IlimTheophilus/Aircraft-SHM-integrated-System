# pages/7_Drone_Vision_HF.py

import streamlit as st
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

st.set_page_config(page_title="Aircraft SHM AI Assistant (HF)", page_icon="✈️")

st.title("🤖 Aircraft SHM AI Assistant (Hugging Face)")
st.write("Type your question about Aircraft SHM:")

# --- SETTINGS ---
HF_API_TOKEN = st.secrets.get("HF_API_KEY")  # your Hugging Face API token
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"  # HF repo name
MODEL_FILE = "mistral-7b-instruct-v0.2.Q2_K.gguf"  # smallest quantized GGUF for CPU
MODEL_DIR = Path("/tmp/hf_models")
MODEL_PATH = MODEL_DIR / MODEL_FILE

# --- CREATE DIR IF MISSING ---
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --- DOWNLOAD MODEL IF NOT EXISTS ---
if not MODEL_PATH.exists():
    st.info("Downloading model (CPU-friendly, ~3GB)... This may take a few minutes.")
    try:
        # Download GGUF file from HF hub
        MODEL_PATH = hf_hub_download(
            repo_id=MODEL_NAME,
            filename=MODEL_FILE,
            cache_dir=str(MODEL_DIR),
            use_auth_token=HF_API_TOKEN
        )
        st.success("Model downloaded successfully!")
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        st.stop()

# --- LOAD MODEL ---
@st.cache_resource
def load_llm():
    return Llama(
        model_path=str(MODEL_PATH),
        n_ctx=2048,       # max tokens for input+output
        n_threads=4,      # adjust based on CPU cores
        n_gpu_layers=0    # CPU only
    )

try:
    llm = load_llm()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# --- CHAT INTERFACE ---
user_input = st.text_input("You:", placeholder="Ask about Aircraft SHM...")

if user_input:
    try:
        # Wrap user input in instruction format
        prompt = f"<s>[INST] {user_input} [/INST]"
        output = llm(prompt, max_tokens=256, echo=False)
        st.markdown(f"**AI:** {output['choices'][0]['text'].strip()}")
    except Exception as e:
        st.error(f"Failed to generate response: {e}")
