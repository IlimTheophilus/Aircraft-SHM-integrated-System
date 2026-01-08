# app/pages/7_Drone_Vision_HF.py
import streamlit as st
from transformers import pipeline
from huggingface_hub import hf_hub_download
from pathlib import Path
import os

st.set_page_config(page_title="Aircraft SHM AI Assistant (HF)", page_icon="🤖")

st.title("🤖 Aircraft SHM AI Assistant (Hugging Face)")
st.markdown("Ask your questions about Aircraft SHM and get AI-generated answers.")

# --- Step 1: Hugging Face API Key ---
try:
    HF_API_TOKEN = st.secrets["HF_API_KEY"]
except KeyError:
    st.error("HF_API_KEY is missing! Add your Hugging Face token in Streamlit Secrets.")
    st.stop()

# --- Step 2: Model Setup ---
MODEL_REPO = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
MODEL_FILE = "mistral-7b-instruct-v0.2.Q3_K_M.gguf"  # Smaller quantized version for CPU

# Path to store downloaded model
MODEL_PATH = Path.home() / "hf_models" / MODEL_FILE
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# Download the model if it doesn't exist
if not MODEL_PATH.exists():
    with st.spinner("Downloading model from Hugging Face (may take a few minutes)..."):
        MODEL_PATH_LOCAL = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            token=HF_API_TOKEN,
            local_dir=str(MODEL_PATH.parent),
            local_dir_use_symlinks=False
        )
        st.success("Model downloaded!")

# --- Step 3: Load text-generation pipeline ---
@st.cache_resource(show_spinner=False)
def load_text_pipeline():
    return pipeline(
        task="text-generation",
        model=str(MODEL_PATH),
        device=-1,  # CPU only
        use_auth_token=HF_API_TOKEN
    )

try:
    text_pipeline = load_text_pipeline()
except Exception as e:
    st.error(f"Failed to load text generation model: {e}")
    st.stop()

# --- Step 4: User input and AI response ---
user_input = st.text_input("Type your question about Aircraft SHM:")

if user_input:
    with st.spinner("Generating AI response..."):
        try:
            response = text_pipeline(user_input, max_new_tokens=128)
            ai_text = response[0]["generated_text"]
            st.markdown(f"**You:** {user_input}")
            st.markdown(f"**AI:** {ai_text}")
        except Exception as e:
            st.error(f"Error generating response: {e}")
