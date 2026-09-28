import streamlit as st
import whisper
import tempfile
import subprocess
import os


# Page configuration
st.set_page_config(
    page_title="Speech-to-Text Application",
    page_icon="🎙️"
)


# Title
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
    type=["wav", "mp3", "m4a", "mpeg", "mpga", "ogg", "flac"]
)


if uploaded_file is not None:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(uploaded_file.name)[1]
    ) as input_file:

        input_file.write(uploaded_file.read())
        input_audio = input_file.name


    # Convert audio to WAV
    output_audio = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ).name

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_audio,
            "-ar",
            "16000",
            "-ac",
            "1",
            output_audio
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )


    # Display audio player
    st.subheader("Audio Input")

    with open(output_audio, "rb") as audio_file:
        audio_bytes = audio_file.read()

    st.audio(audio_bytes, format="audio/wav")


    # Transcription button
    if st.button("Transcribe Audio"):

        with st.spinner("Transcribing audio..."):

            result = model.transcribe(output_audio)

        st.success("Transcription completed!")

        st.subheader("Transcribed Text")

        st.write(result["text"].strip())


    # Delete temporary files
    if os.path.exists(input_audio):
        os.remove(input_audio)

    if os.path.exists(output_audio):
        os.remove(output_audio)