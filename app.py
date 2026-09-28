import streamlit as st
import whisper
import tempfile
import os


# Page configuration
st.set_page_config(
    page_title="Speech-to-Text Application",
    page_icon="🎙️"
)


# Application title
st.title("🎙️ Speech-to-Text Application")
st.write(
    "Convert recorded speech into text using the pre-trained Whisper model."
)


# Load Whisper model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")


model = load_whisper_model()


# Upload audio file
uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3", "m4a", "ogg", "flac"]
)


# Process uploaded audio
if uploaded_file is not None:

    st.audio(uploaded_file)

    if st.button("Transcribe Audio"):

        # Save uploaded audio temporarily
        file_extension = os.path.splitext(uploaded_file.name)[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension
        ) as temp_audio:

            temp_audio.write(uploaded_file.read())
            audio_file = temp_audio.name

        # Transcribe audio
        with st.spinner("Transcribing audio..."):

            result = model.transcribe(audio_file)

        # Display result
        st.success("Transcription completed!")

        st.subheader("Transcribed Text")
        st.write(result["text"].strip())

        # Remove temporary file
        os.remove(audio_file)