# pages/7_Drone_Vision_AI_Assistant_HF.py

import streamlit as st
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from pathlib import Path

st.set_page_config(page_title="Aircraft SHM AI Assistant (HF)", layout="wide")
st.title("🤖 Aircraft SHM AI Assistant (Hugging Face)")

# --- Hugging Face model settings ---
HF_API_TOKEN = st.secrets["HF_API_KEY"]  # Make sure you saved this in Streamlit secrets
HF_MODEL_REPO = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
HF_MODEL_FILE = "mistral-7b-instruct-v0.2.Q2_K.gguf"  # Smallest CPU-friendly (~3GB)
LOCAL_MODEL_PATH = Path.home() / "hf_models" / HF_MODEL_FILE
LOCAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- Function to load or download model ---
@st.cache_resource
def load_model():
    if not LOCAL_MODEL_PATH.exists():
        try:
            st.info("Downloading model... this may take a few minutes!")
            hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=HF_MODEL_FILE,
                cache_dir=str(LOCAL_MODEL_PATH.parent),
                use_auth_token=HF_API_TOKEN
            )
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            st.stop()

    st.success("Loading model into memory...")
    llm = Llama(
        model_path=str(LOCAL_MODEL_PATH),
        n_ctx=2048,        # context length
        n_threads=4,       # adjust for CPU cores
        n_gpu_layers=0     # change if GPU available
    )
    st.success("Model loaded successfully!")
    return llm

# --- Load model ---
llm = load_model()

# --- Session state for chat history ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Chat interface ---
user_input = st.text_input("Type your question about Aircraft SHM:")

if st.button("Send") and user_input.strip():
    st.session_state.history.append({"role": "user", "content": user_input})

    # Prepare prompt for LLaMA-style instruct model
    prompt = "<s>[INST] " + user_input + " [/INST]"
    try:
        output = llm(prompt, max_tokens=512, stop=["</s>"], echo=False)
        response = output["choices"][0]["text"].strip()
    except Exception as e:
        response = f"⚠️ Model inference error: {e}"

    st.session_state.history.append({"role": "assistant", "content": response})

# --- Display chat history ---
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**AI:** {msg['content']}")
