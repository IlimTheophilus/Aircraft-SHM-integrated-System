# pages/7_Drone_Vision_HF_AI.py
import streamlit as st
from huggingface_hub import hf_hub_download
from transformers import pipeline
from llama_cpp import Llama

st.set_page_config(
    page_title="Aircraft SHM AI Assistant (HF)",
    page_icon="✈️",
    layout="wide"
)

st.title("🤖 Aircraft SHM AI Assistant (Hugging Face)")

# --- Load your Hugging Face API token from Streamlit secrets ---
HF_API_TOKEN = st.secrets.get("HF_API_KEY")
if HF_API_TOKEN is None:
    st.error("HF_API_KEY not found in Streamlit secrets. Please add it first.")
    st.stop()

# --- Load model with caching to avoid repeated downloads ---
@st.cache_resource(show_spinner=True)
def load_model():
    try:
        # Automatically download GGUF model from HF Hub
        model_path = hf_hub_download(
            repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            use_auth_token=HF_API_TOKEN
        )
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        st.stop()

    # Load the model using llama_cpp
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,       # max tokens in context
        n_threads=4,      # adjust according to your server
        n_gpu_layers=0    # set >0 if GPU available
    )
    return llm

# --- Initialize the model ---
llm = load_model()

# --- Chat UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your question about Aircraft SHM:")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Prepare prompt for Mistral Instruct format
    prompt = "<s>[INST] " + user_input + " [/INST]"

    # Generate AI response
    try:
        output = llm(prompt, max_tokens=256, stop=["</s>"], echo=False)
        ai_response = output['choices'][0]['text'] if 'choices' in output else output['text']
    except Exception as e:
        ai_response = f"⚠️ Failed to generate response: {e}"

    # Save AI response
    st.session_state.messages.append({"role": "assistant", "content": ai_response})

# --- Display conversation ---
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**AI:** {msg['content']}")
