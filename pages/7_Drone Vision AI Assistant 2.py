import streamlit as st
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers import pipeline

st.set_page_config(page_title="Aircraft SHM AI Assistant (HF)", layout="wide")

st.title("🤖 Aircraft SHM AI Assistant (Hugging Face)")

# ---- Hugging Face API Token ----
HF_API_TOKEN = st.secrets["HF_API_KEY"]

# ---- Model Setup ----
@st.cache_resource(show_spinner=True)
def load_models():
    """
    Downloads and loads the Hugging Face text generation and TTS models.
    Returns:
        text_pipeline: Hugging Face text generation pipeline
        tts_pipeline: Hugging Face text-to-speech pipeline
    """
    # Model names
    HF_TEXT_MODEL = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    HF_TTS_MODEL = "facebook/fastspeech2-en-ljspeech"

    # Download GGUF file for text generation
    model_dir = Path.home() / "hf_models" / "mistral"
    model_dir.mkdir(parents=True, exist_ok=True)
    GGUF_FILE = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    model_path = hf_hub_download(
        repo_id=HF_TEXT_MODEL,
        filename=GGUF_FILE,
        token=HF_API_TOKEN,
        cache_dir=model_dir
    )

    # Text generation pipeline
    text_pipeline = pipeline(
        task="text-generation",
        model=model_path,
        device=0 if st.runtime.exists("gpu") else -1  # use GPU if available
    )

    # Text-to-speech pipeline
    tts_pipeline = pipeline(
        task="text-to-speech",
        model=HF_TTS_MODEL,
        use_auth_token=HF_API_TOKEN
    )

    return text_pipeline, tts_pipeline

# Load models
text_pipeline, tts_pipeline = load_models()

# ---- User Input ----
user_input = st.text_input("Type your question about Aircraft SHM:")

if user_input:
    with st.spinner("Generating AI response..."):
        # Text generation
        response = text_pipeline(user_input, max_new_tokens=256, do_sample=True)
        ai_text = response[0]["generated_text"]
        st.markdown(f"**You:** {user_input}")
        st.markdown(f"**AI:** {ai_text}")

        # Text-to-speech
        try:
            audio_path = Path("tts_output.wav")
            tts_pipeline(ai_text, save_to_file=audio_path)
            st.audio(str(audio_path))
        except Exception as e:
            st.error(f"⚠️ Hugging Face TTS error: {e}")
