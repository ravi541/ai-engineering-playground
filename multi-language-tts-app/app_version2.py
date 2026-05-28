import ollama
import streamlit as st
from gtts import gTTS
from deep_translator import GoogleTranslator
import uuid

# ------------------------------------------------
# STREAMLIT PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="AI Multilingual Voice Assistant",
    page_icon="🤖",
    layout="wide"
)

# ------------------------------------------------
# SESSION STATE
# ------------------------------------------------

# Conversation ID
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Latest AI response
if "latest_response" not in st.session_state:
    st.session_state.latest_response = ""

# ------------------------------------------------
# TITLE
# ------------------------------------------------

st.title("🤖 AI Multilingual Voice Assistant")

st.write(f"### 🆔 Conversation ID: {st.session_state.conversation_id}")

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

st.sidebar.title("⚙️ Controls")

# New chat button
if st.sidebar.button("🆕 New Conversation"):

    st.session_state.conversation_id = str(uuid.uuid4())
    st.session_state.chat_history = []
    st.session_state.latest_response = ""

    st.rerun()

# ------------------------------------------------
# LANGUAGE OPTIONS
# ------------------------------------------------

language_options = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "Bengali": "bn",
    "Urdu": "ur",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh-cn",
    "Russian": "ru",
    "Arabic": "ar"
}

selected_language = st.sidebar.selectbox(
    "🌍 Select Response Language",
    language_options.keys()
)

# ------------------------------------------------
# USER INPUT
# ------------------------------------------------

text = st.chat_input("Ask something...")

# ------------------------------------------------
# AI RESPONSE GENERATION
# ------------------------------------------------

if text:

    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": text
    })

    with st.spinner("Generating AI response..."):

        response = ollama.chat(
            model='gemma3:270m',
            messages=st.session_state.chat_history
        )

        ai_response = response['message']['content']

        # Save latest response
        st.session_state.latest_response = ai_response

        # Add assistant response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": ai_response
        })

# ------------------------------------------------
# DISPLAY CHAT HISTORY
# ------------------------------------------------

st.header("💬 Conversation")

for message in st.session_state.chat_history:

    if message["role"] == "user":

        with st.chat_message("user"):
            st.write(message["content"])

    else:

        with st.chat_message("assistant"):
            st.write(message["content"])

# ------------------------------------------------
# TEXT TO SPEECH
# ------------------------------------------------

if st.session_state.latest_response:

    st.divider()

    st.subheader("🔊 Voice Output")

    if st.button("🎤 Generate Voice Response"):

        lang_code = language_options[selected_language]

        # ----------------------------------------
        # TRANSLATE RESPONSE
        # ----------------------------------------

        if lang_code != "en":

            translated_text = GoogleTranslator(
                source='auto',
                target=lang_code
            ).translate(st.session_state.latest_response)

        else:
            translated_text = st.session_state.latest_response

        st.subheader("🌍 Translated Response")

        st.write(translated_text)

        # ----------------------------------------
        # GENERATE SPEECH
        # ----------------------------------------

        tts = gTTS(
            text=translated_text,
            lang=lang_code
        )

        audio_file = "response.mp3"

        tts.save(audio_file)

        # ----------------------------------------
        # PLAY AUDIO
        # ----------------------------------------

        audio_bytes = open(audio_file, "rb").read()

        st.audio(
            audio_bytes,
            format="audio/mp3"
        )

        st.success("✅ Voice generated successfully!")

# ------------------------------------------------
# SIDEBAR HISTORY
# ------------------------------------------------

st.sidebar.divider()

st.sidebar.subheader("📝 Chat History")

for index, msg in enumerate(st.session_state.chat_history):

    role = msg["role"]

    content = msg["content"][:40]

    st.sidebar.write(f"{index+1}. {role}: {content}...")