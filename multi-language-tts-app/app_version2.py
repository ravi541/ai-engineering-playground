import ollama
import streamlit as st
from gtts import gTTS
from deep_translator import GoogleTranslator
import uuid
import speech_recognition as sr

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="AI Multilingual Voice Assistant",
    page_icon="🤖",
    layout="wide"
)

# ------------------------------------------------
# SESSION STATE
# ------------------------------------------------

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "latest_response" not in st.session_state:
    st.session_state.latest_response = ""

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# ------------------------------------------------
# TITLE
# ------------------------------------------------

st.title("🤖 AI Multilingual Voice Assistant")

st.write(
    f"### 🆔 Conversation ID: {st.session_state.conversation_id}"
)

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

st.sidebar.title("⚙️ Controls")

# ------------------------------------------------
# NEW CHAT
# ------------------------------------------------

if st.sidebar.button("🆕 New Conversation"):

    st.session_state.conversation_id = str(uuid.uuid4())
    st.session_state.chat_history = []
    st.session_state.latest_response = ""
    st.session_state.input_text = ""

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
# SPEECH TO TEXT
# ------------------------------------------------

st.sidebar.divider()

st.sidebar.subheader("🎤 Voice Input")

if st.sidebar.button("🎙️ Start Listening"):

    recognizer = sr.Recognizer()

    try:

        with sr.Microphone() as source:

            st.sidebar.info("🎤 Listening... Speak now")

            audio = recognizer.listen(source)

            st.sidebar.info("🧠 Recognizing speech...")

            voice_text = recognizer.recognize_google(audio)

            st.sidebar.success(
                "✅ Voice captured successfully!"
            )

            st.sidebar.write(
                f"🗣️ You said: {voice_text}"
            )

            # Save into session state
            st.session_state.input_text = voice_text

            # Rerun app
            st.rerun()

    except Exception as e:

        st.sidebar.error(f"❌ Error: {str(e)}")

# ------------------------------------------------
# USER INPUT
# ------------------------------------------------

text = st.chat_input("Ask something...")

# Voice input processing
if st.session_state.input_text:

    text = st.session_state.input_text

    st.session_state.input_text = ""

# ------------------------------------------------
# AI RESPONSE GENERATION
# ------------------------------------------------

if text:

    # Add user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": text
    })

    with st.spinner("🤖 Generating AI response..."):

        response = ollama.chat(
            model='gemma3:270m',
            messages=st.session_state.chat_history
        )

        ai_response = response['message']['content']

        # Save latest response
        st.session_state.latest_response = ai_response

        # Add assistant response
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
            ).translate(
                st.session_state.latest_response
            )

        else:

            translated_text = (
                st.session_state.latest_response
            )

        # ----------------------------------------
        # SHOW TRANSLATED RESPONSE
        # ----------------------------------------

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

        audio_bytes = open(
            audio_file,
            "rb"
        ).read()

        st.audio(
            audio_bytes,
            format="audio/mp3"
        )

        st.success(
            "✅ Voice generated successfully!"
        )

# ------------------------------------------------
# SIDEBAR HISTORY
# ------------------------------------------------

st.sidebar.divider()

st.sidebar.subheader("📝 Chat History")

for index, msg in enumerate(
    st.session_state.chat_history
):

    role = msg["role"]

    content = msg["content"][:40]

    st.sidebar.write(
        f"{index+1}. {role}: {content}..."
    )