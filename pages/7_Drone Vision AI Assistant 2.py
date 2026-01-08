import streamlit as st
from openai import OpenAI, RateLimitError
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time
import pyttsx3  # For text-to-speech (local)
import speech_recognition as sr  # For voice input

# --------------------------
# SETTINGS
# --------------------------
MAX_MESSAGES = 8
MAX_TOKENS = 300
OPENAI_MODEL = "gpt-4o-mini"
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# --------------------------
# INITIALIZE SESSION STATE
# --------------------------
if "ai_messages" not in st.session_state:
    st.session_state.ai_messages = []

if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))

# --------------------------
# VOICE INPUT FUNCTION
# --------------------------
def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now!")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.success(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Could not understand audio.")
            return ""
        except sr.RequestError:
            st.error("Speech Recognition service failed.")
            return ""

# --------------------------
# VOICE OUTPUT FUNCTION
# --------------------------
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# --------------------------
# OPENAI RESPONSE FUNCTION
# --------------------------
def get_openai_response(messages):
    # Limit history
    if len(messages) > MAX_MESSAGES + 1:
        messages = [messages[0]] + messages[-MAX_MESSAGES:]
    try:
        time.sleep(1)  # Throttle API calls
        response = st.session_state.openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS
        )
        return response.choices[0].message.content
    except RateLimitError:
        return "⚠️ OpenAI service busy. Try again in a few seconds."

# --------------------------
# HUGGING FACE RESPONSE FUNCTION
# --------------------------
@st.cache_resource
def load_hf_model():
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL)
    hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return hf_pipeline

hf_pipeline = load_hf_model()

def get_hf_response(prompt):
    output = hf_pipeline(prompt, max_length=MAX_TOKENS, do_sample=True)
    return output[0]["generated_text"]

# --------------------------
# STREAMLIT UI
# --------------------------
st.title("🤖 Aircraft SHM AI Assistant")
st.markdown("Ask questions about Aircraft Structural Health Monitoring (SHM). Supports voice input and chat history.")

# Voice input button
if st.button("🎤 Speak"):
    voice_text = get_voice_input()
    if voice_text:
        st.session_state.ai_messages.append({"role": "user", "content": voice_text})
        # Get AI response
        openai_resp = get_openai_response(st.session_state.ai_messages)
        hf_resp = get_hf_response(voice_text)
        combined_resp = f"**OpenAI:** {openai_resp}\n\n**HF:** {hf_resp}"
        st.session_state.ai_messages.append({"role": "assistant", "content": combined_resp})
        speak_text(combined_resp)

# Text input
user_input = st.text_input("Or type your question here:")
if st.button("Send") and user_input:
    st.session_state.ai_messages.append({"role": "user", "content": user_input})
    # Get AI response
    openai_resp = get_openai_response(st.session_state.ai_messages)
    hf_resp = get_hf_response(user_input)
    combined_resp = f"**OpenAI:** {openai_resp}\n\n**HF:** {hf_resp}"
    st.session_state.ai_messages.append({"role": "assistant", "content": combined_resp})
    speak_text(combined_resp)

# Display chat
for msg in st.session_state.ai_messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**AI:** {msg['content']}")
