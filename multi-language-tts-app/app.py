import streamlit as st
from gtts import gTTS
from deep_translator import GoogleTranslator
from langdetect import detect
from textblob import TextBlob
import spacy
import os

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Streamlit page settings
st.set_page_config(
    page_title="AI Multilingual NLP Assistant",
    page_icon="🧠",
    layout="centered"
)

# Title
st.title("🧠 AI Multilingual NLP Assistant")

st.write("""
This app performs:
- NLP Analysis
- Tokenization
- Lemmatization
- POS Tagging
- Named Entity Recognition
- Stopword Detection
- Sentiment Analysis
- Text-to-Speech
""")

# User input
text = st.text_area("✍️ Enter Text Here")

# All major world languages supported by gTTS

language_options = {
    "Afrikaans": "af",
    "Arabic": "ar",
    "Bulgarian": "bg",
    "Bengali": "bn",
    "Bosnian": "bs",
    "Catalan": "ca",
    "Czech": "cs",
    "Welsh": "cy",
    "Danish": "da",
    "German": "de",
    "Greek": "el",
    "English": "en",
    "Esperanto": "eo",
    "Spanish": "es",
    "Estonian": "et",
    "Finnish": "fi",
    "French": "fr",
    "Gujarati": "gu",
    "Hindi": "hi",
    "Croatian": "hr",
    "Hungarian": "hu",
    "Indonesian": "id",
    "Icelandic": "is",
    "Italian": "it",
    "Hebrew": "iw",
    "Japanese": "ja",
    "Javanese": "jw",
    "Khmer": "km",
    "Kannada": "kn",
    "Korean": "ko",
    "Latin": "la",
    "Lithuanian": "lt",
    "Latvian": "lv",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Malay": "ms",
    "Myanmar": "my",
    "Nepali": "ne",
    "Dutch": "nl",
    "Norwegian": "no",
    "Punjabi": "pa",
    "Polish": "pl",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Sinhala": "si",
    "Slovak": "sk",
    "Albanian": "sq",
    "Serbian": "sr",
    "Sundanese": "su",
    "Swedish": "sv",
    "Swahili": "sw",
    "Tamil": "ta",
    "Telugu": "te",
    "Thai": "th",
    "Filipino": "tl",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Vietnamese": "vi",
    "Chinese (Simplified)": "zh-cn",
    "Chinese (Traditional)": "zh-tw"
}

selected_language = st.selectbox(
    "🌍 Select Output Language",
    sorted(language_options.keys())
)

# NLP Processing
if text:

    # Process text using spaCy
    doc = nlp(text)

    st.header("🧠 NLP Analysis")

    # ------------------------------------------------
    # TOKENS
    # ------------------------------------------------
    tokens = [token.text for token in doc]

    st.subheader("1️⃣ Tokens")
    st.write(tokens)

    # ------------------------------------------------
    # LEMMAS
    # ------------------------------------------------
    lemmas = [token.lemma_ for token in doc]

    st.subheader("2️⃣ Lemmas")
    st.write(lemmas)

    # ------------------------------------------------
    # POS TAGGING
    # ------------------------------------------------
    st.subheader("3️⃣ POS Tagging")

    pos_tags = []

    for token in doc:
        pos_tags.append(
            {
                "Word": token.text,
                "POS": token.pos_,
                "Tag": token.tag_
            }
        )

    st.write(pos_tags)

    # ------------------------------------------------
    # NER
    # ------------------------------------------------
    st.subheader("4️⃣ Named Entities")

    entities = []

    for ent in doc.ents:
        entities.append(
            {
                "Entity": ent.text,
                "Label": ent.label_
            }
        )

    if entities:
        st.write(entities)
    else:
        st.write("No entities detected")

    # ------------------------------------------------
    # STOPWORDS
    # ------------------------------------------------
    st.subheader("5️⃣ Stopwords")

    stop_words = [token.text for token in doc if token.is_stop]

    st.write(stop_words)

    # ------------------------------------------------
    # LANGUAGE DETECTION
    # ------------------------------------------------
    st.subheader("6️⃣ Language Detection")

    detected_language = detect(text)

    st.success(f"Detected Language: {detected_language}")

    # ------------------------------------------------
    # SENTIMENT ANALYSIS
    # ------------------------------------------------
    st.subheader("7️⃣ Sentiment Analysis")

    blob = TextBlob(text)

    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    st.write(f"Polarity: {polarity}")
    st.write(f"Subjectivity: {subjectivity}")

    # Sentiment Label
    if polarity > 0:
        st.success("😊 Positive Sentiment")

    elif polarity < 0:
        st.error("😞 Negative Sentiment")

    else:
        st.info("😐 Neutral Sentiment")

    # ------------------------------------------------
    # TOKEN DETAILS
    # ------------------------------------------------
    st.subheader("8️⃣ Detailed Token Analysis")

    for token in doc:

        st.write(
            f"""
            Word: {token.text}

            POS: {token.pos_}

            Lemma: {token.lemma_}

            Dependency: {token.dep_}

            Shape: {token.shape_}

            Is Alpha: {token.is_alpha}

            Is Stopword: {token.is_stop}

            ---------------------------
            """
        )

# ------------------------------------------------
# TEXT TO SPEECH
# ------------------------------------------------

# ------------------------------------------------
# TEXT TO SPEECH
# ------------------------------------------------

if st.button("🔊 Convert To Speech"):

    if text.strip() == "":
        st.warning("Please enter some text")

    else:

        lang_code = language_options[selected_language]

        # ----------------------------------------
        # Translate text if needed
        # ----------------------------------------

        if lang_code != "en":

            translated_text = GoogleTranslator(
                source='auto',
                target=lang_code
            ).translate(text)

        else:
            translated_text = text

        # Show translated text
        st.subheader("🌍 Translated Text")
        st.write(translated_text)

        # ----------------------------------------
        # Create Speech
        # ----------------------------------------

        tts = gTTS(
            text=translated_text,
            lang=lang_code
        )

        # Save audio
        audio_file = "output.mp3"

        tts.save(audio_file)

        # Play audio
        audio_bytes = open(audio_file, "rb").read()

        st.audio(
            audio_bytes,
            format="audio/mp3"
        )

        st.success("✅ Speech generated successfully!")