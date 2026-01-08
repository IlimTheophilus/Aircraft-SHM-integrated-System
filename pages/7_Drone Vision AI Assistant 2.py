import streamlit as st
from pathlib import Path
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# ------------------------------
# Streamlit app title
# ------------------------------
st.set_page_config(page_title="Aircraft SHM AI Assistant (HF)", layout="wide")
st.title("🤖 Aircraft SHM AI Assistant (Hugging Face)")
st.write("Ask questions about Aircraft SHM. Powered by Mistral 7B Instruct (GGUF).")

# ------------------------------
# Hugging Face API Token & Model
# ------------------------------
HF_API_TOKEN = st.secrets["HF_API_KEY"]  # Add this in your Streamlit secrets
HF_MODEL_REPO = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
HF_MODEL_FILE = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"  # Correct working GGUF
LOCAL_MODEL_PATH = Path.home() / "hf_models" / HF_MODEL_FILE
LOCAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# ------------------------------
# Load LLM
# ------------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    # Download if not exists
    if not LOCAL_MODEL_PATH.exists():
        st.info("Downloading model (~4.37 GB). This may take a while...")
        try:
            hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=HF_MODEL_FILE,
                cache_dir=str(LOCAL_MODEL_PATH.parent),
                use_auth_token=HF_API_TOKEN
            )
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            st.stop()

    # Load model
    st.info("Loading model into memory...")
    llm = Llama(
        model_path=str(LOCAL_MODEL_PATH),
        n_ctx=2048,       # max sequence length
        n_threads=4,      # adjust based on your CPU
        n_gpu_layers=0    # set >0 if GPU available
    )
    st.success("Model loaded!")
    return llm

llm = load_model()

# ------------------------------
# User input
# ------------------------------
user_input = st.text_input("Type your question about Aircraft SHM:")

if user_input:
    with st.spinner("Generating response..."):
        # Llama expects <s>[INST] ... [/INST] format for instruction models
        prompt = f"<s>[INST] {user_input} [/INST]"
        output = llm(prompt, max_tokens=512, stop=["</s>"], echo=False)
        response_text = output["choices"][0]["text"]
        st.markdown(f"**You:** {user_input}")
        st.markdown(f"**AI:** {response_text}")
