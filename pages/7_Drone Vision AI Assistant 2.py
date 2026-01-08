# app/pages/7_Drone_Vision_HF.py
import streamlit as st
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from pathlib import Path

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
MODEL_FILE = "mistral-7b-instruct-v0.2.Q3_K_M.gguf"  # smaller, CPU-friendly
MODEL_PATH = Path.home() / "hf_models" / MODEL_FILE
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# Download GGUF model if missing
if not MODEL_PATH.exists():
    with st.spinner("Downloading model (may take a few minutes)..."):
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            token=HF_API_TOKEN,
            local_dir=str(MODEL_PATH.parent),
            local_dir_use_symlinks=False
        )
        st.success("Model downloaded!")

# --- Step 3: Load LLaMA GGUF model ---
@st.cache_resource(show_spinner=False)
def load_llama_model():
    return Llama(
        model_path=str(MODEL_PATH),
        n_ctx=2048,        # context length
        n_threads=4,       # CPU threads
        n_gpu_layers=0     # CPU-only
    )

try:
    llm = load_llama_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# --- Step 4: User input and AI response ---
user_input = st.text_input("Type your question about Aircraft SHM:")

if user_input:
    with st.spinner("Generating AI response..."):
        try:
            prompt = f"<s>[INST] {user_input} [/INST]"
            output = llm(prompt, max_tokens=128, stop=["</s>"], echo=False)
            st.markdown(f"**You:** {user_input}")
            st.markdown(f"**AI:** {output['choices'][0]['text']}")
        except Exception as e:
            st.error(f"Error generating response: {e}")
