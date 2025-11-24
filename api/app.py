import os
import io
import tempfile
import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------
# Setup OpenAI client for TTS
# ---------------------------
# Load API key
# ---------------------------
# Setup OpenAI client for TTS
# ---------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def text_to_speech_file(message: str, model_name: str) -> str:
    """
    Convert text to speech using OpenAI TTS and save it to a temp MP3 file.
    Returns the file path.
    """
    if not message or not message.strip():
        return ""

    # 1. Call TTS (same logic as your working test.py)
    resp = client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=message
    )

    # 2. Collect streaming bytes
    audio_buf = io.BytesIO()
    for chunk in resp.iter_bytes():
        audio_buf.write(chunk)
    audio_buf.seek(0)

    # 3. Write to a temporary mp3 file
    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=f"_{model_name}.mp3"
    )
    tmp.write(audio_buf.read())
    tmp.flush()
    tmp.close()

    return tmp.name  # path to MP3

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(
    page_title="Multi-Model Meta Q&A",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 FuseFind")
st.caption(
    "Ask any question and compare answers from OpenAI, Claude, Groq, and Ollama — with audio playback."
)

# ---------------------------
# Session state init
# ---------------------------
if "answers" not in st.session_state:
    st.session_state["answers"] = None
if "last_question" not in st.session_state:
    st.session_state["last_question"] = ""

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("⚙️ Settings")

use_openai = st.sidebar.checkbox("OpenAI", value=True)
use_claude = st.sidebar.checkbox("Claude", value=True)
use_groq = st.sidebar.checkbox("Groq", value=True)
use_ollama = st.sidebar.checkbox("Ollama (local)", value=True)

temperature = st.sidebar.slider(
    "Creativity (temperature UI only)", 0.0, 1.0, 0.3, 0.1
)
st.sidebar.caption(
    "Temperature is not sent to backend yet – backend uses fixed values."
)

# ---------------------------
# Backend call helper
# ---------------------------
BACKEND_URL = "http://localhost:8000/meta-qa"


def get_meta_answers(question: str) -> dict:
    resp = requests.post(
        BACKEND_URL,
        json={"question": question},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()  # {"openai": "...", "claude": "...", ...}


# ---------------------------
# Main input area
# ---------------------------
st.markdown("### 💬 Ask your question")

question = st.text_area(
    "Type your question here:",
    placeholder="e.g. Explain machine learning in simple terms.",
    height=120,
    label_visibility="collapsed",
    value=st.session_state.get("last_question", ""),
)

col_btn, col_info = st.columns([1, 3])
with col_btn:
    ask_button = st.button("🚀 Ask all models", use_container_width=True)

with col_info:
    st.markdown(
        "<div style='margin-top: 0.4rem; font-size: 0.9rem; color: gray;'>"
        "Tip: Enable/disable models from the sidebar to focus on specific engines."
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ---------------------------
# When user clicks 'Ask all models' -> call backend ONCE
# and store results in session_state
# ---------------------------
if ask_button and question.strip():
    try:
        with st.spinner("Querying all models..."):
            answers = get_meta_answers(question.strip())
        st.session_state["answers"] = answers
        st.session_state["last_question"] = question.strip()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling backend: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

elif ask_button and not question.strip():
    st.warning("Please enter a question before clicking 'Ask all models'.")

# ---------------------------
# Display answers from session_state (persists across reruns)
# ---------------------------
answers = st.session_state.get("answers")

if answers:
    # Filter models based on sidebar
    model_order = []
    if use_openai and "openai" in answers:
        model_order.append("openai")
    if use_claude and "claude" in answers:
        model_order.append("claude")
    if use_groq and "groq" in answers:
        model_order.append("groq")
    if use_ollama and "ollama" in answers:
        model_order.append("ollama")

    if not model_order:
        st.warning("No models selected. Please enable at least one model in the sidebar.")
    else:
        tab_labels = [name.upper() for name in model_order]
        tabs = st.tabs(tab_labels)

        for name, tab in zip(model_order, tabs):
            with tab:
                st.markdown(f"#### 🧠 {name.upper()} Response")
                answer_text = answers.get(name, "[no response]")

                is_error = isinstance(answer_text, str) and answer_text.startswith(
                    f"[{name} error"
                )

                if is_error:
                    st.error(answer_text)
                else:
                    st.write(answer_text)

                    # Play audio button (this causes a rerun, but answers are kept in session_state)
                    if st.button(f"🔊 Play {name.upper()} audio", key=f"play_{name}"):
                        with st.spinner("Generating audio with OpenAI TTS..."):
                            audio_path = text_to_speech_file(answer_text, name)

                        if audio_path and os.path.exists(audio_path):
                            st.write(f"🎧 Generated audio file: `{os.path.basename(audio_path)}`")  # optional debug
                            st.audio(audio_path)  # Streamlit will detect it's an mp3 file
                        else:
                            st.error("Audio generation failed or returned an empty file.")

# ---------------------------
# Notes
# ---------------------------
with st.expander("🔍 Notes"):
    st.write(
        "Audio uses OpenAI TTS (`tts-1`, voice `onyx`) for all models. "
        "Text still comes from each respective model (OpenAI, Claude, Groq, Ollama).\n\n"
        "Buttons in Streamlit rerun the whole script, so we store the latest answers in "
        "`st.session_state` to keep them visible when you click 'Play audio'."
    )
