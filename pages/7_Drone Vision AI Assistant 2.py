import streamlit as st
from transformers import pipeline

# Get your Hugging Face API token
HF_API_TOKEN = st.secrets["HF_API_KEY"]  # or set via env var

st.title("Aircraft SHM AI Assistant (HF API)")

# Create a remote pipeline
@st.cache_resource
def load_pipeline():
    return pipeline(
        "text-generation",
        model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        use_auth_token=HF_API_TOKEN
    )

generator = load_pipeline()

# Chat input
user_input = st.text_input("Type your question about Aircraft SHM:")

if user_input:
    with st.spinner("Generating response..."):
        output = generator(user_input, max_new_tokens=200, do_sample=True)
        st.write("AI:", output[0]["generated_text"])
