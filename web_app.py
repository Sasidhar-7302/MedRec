
import streamlit as st
import logging
import tempfile
import time
import os
from pathlib import Path
from audiorecorder import audiorecorder
import soundfile as sf
import sounddevice as sd
import numpy as np

# Internal App Imports
from app.config import load_config
from app.transcriber import WhisperTranscriber
from app.two_pass_summarizer import TwoPassSummarizer
from app.doctor_assistant import DoctorAssistant
from app.doctor_profiles import DoctorProfileManager

# --- Page Configuration ---
st.set_page_config(
    page_title="GI Scribe - Gastroenterology Dictation",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS Loading ---
def load_css():
    css_path = Path("app/web_styles.css")
    if css_path.exists():
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# --- Initialize Session State ---
if "app_config" not in st.session_state:
    st.session_state.app_config = load_config()

# --- Cached Model Loading ---
@st.cache_resource
def get_backend():
    """Load and cache the backend models to prevent reloading on every run."""
    config = load_config()
    transcriber = WhisperTranscriber(config.whisper)
    summarizer = TwoPassSummarizer(config.summarizer)
    return transcriber, summarizer

st.session_state.transcriber, st.session_state.summarizer = get_backend()

if "transcript" not in st.session_state:
    st.session_state.transcript = ""
    st.session_state.summary = ""
    st.session_state.last_audio_path = None
    st.session_state.processing_complete = False

# --- Sidebar ---
with st.sidebar:
    st.title("GI Scribe")
    st.markdown("### Gastroenterology Dictation")
    st.markdown("---")
    
    page = st.radio("Navigation", ["🎙️ Record", "📂 History", "⚙️ Settings"])
    
    st.markdown("---")
    st.subheader("System Status")
    st.success("Whisper: Ready")
    st.success("Ollama: Ready")
    
    st.markdown("---")
    st.caption("GI Specialists of Georgia")

# --- Helper Functions ---
def process_audio(audio_path, original_dialogue=None):
    """Transcribes, diarizes, and summarizes the audio."""
    with st.spinner("🎧 Transcribing audio with Whisper..."):
        try:
            result = st.session_state.transcriber.transcribe(Path(audio_path))
            st.session_state.raw_transcript = result.text
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            return

    with st.spinner("🗣️ Separating speakers (Diarization)..."):
        try:
            st.session_state.transcript = st.session_state.summarizer.diarize(st.session_state.raw_transcript)
        except Exception as e:
            st.warning(f"Diarization failed, falling back to raw: {e}")
            st.session_state.transcript = st.session_state.raw_transcript

    with st.spinner("🤖 Generating Clinical Summary..."):
        try:
            summary_text = st.session_state.summarizer.summarize_text(st.session_state.raw_transcript)
            st.session_state.summary = summary_text
            st.session_state.processing_complete = True
            st.session_state.original_dialogue = original_dialogue
        except Exception as e:
            st.error(f"Summarization failed: {e}")

# --- Main Content ---
if page == "🎙️ Record":
    st.markdown("## 🎙️ New Dictation")
    
    if "raw_transcript" not in st.session_state:
        st.session_state.raw_transcript = ""
    if "original_dialogue" not in st.session_state:
        st.session_state.original_dialogue = None
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.info("Click the microphone to start recording. Click again to stop.")
        audio_data = audiorecorder("Start Recording", "Stop Recording")
        
        st.markdown("---")
        st.markdown("### 📂 Upload Audio")
        uploaded_file = st.file_uploader("Upload a WAV or MP3 file", type=["wav", "mp3"])
        
        if uploaded_file is not None:
             # Save upload to temp
             with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as fp:
                 fp.write(uploaded_file.getvalue())
                 st.session_state.last_audio_path = fp.name
                 
             st.audio(st.session_state.last_audio_path)
             if st.button("📝 Process Uploaded File", type="primary"):
                 process_audio(st.session_state.last_audio_path)

        
        # Load Dialogue Button (Kaggle)
        st.markdown("### 📚 Demo Data")
        if st.button("Load Kaggle Case (RUQ Pain / Biliary Colic)"):
            dialogue_path = Path("kaggle_dialogue.mp3")
            if dialogue_path.exists():
                st.session_state.last_audio_path = str(dialogue_path)
                # Load original text for comparison (from generate_fast_audio or known source)
                # For demo, we know Case 2 text. Better to have it accessible.
                # I'll just hardcode the reference for the demo button.
                original_text = (
                    "D: I was wondering if you could tell me a little bit about what brought you in to the Emergency Department today?\n"
                    "P: Yeah, so nice to meet you. I've been having this pain right in my abdomen..."
                )
                process_audio(dialogue_path, original_dialogue=original_text)
            else:
                st.warning("Dialogue audio not found. Please wait for generation.")

        if audio_data is not None and len(audio_data) > 0:
            # Save the recorded audio to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
                # audio_data is a pydub AudioSegment
                audio_data.export(fp.name, format="wav")
                st.session_state.last_audio_path = fp.name
                
            st.audio(st.session_state.last_audio_path, format="audio/wav")
            
            if st.button("📝 Process Recording", type="primary"):
                process_audio(st.session_state.last_audio_path)

    
    # Results Display
    if st.session_state.transcript or st.session_state.summary:
        st.markdown("---")
        
        if st.session_state.original_dialogue:
            st.markdown("### 📊 Kaggle Dataset Comparison")
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                st.info("**Reference Dialogue (Original Kaggle Text)**")
                st.code(st.session_state.original_dialogue, language="text")
            with comp_col2:
                st.success("**AI Diarized Transcription (From Audio)**")
                st.code(st.session_state.transcript, language="text")
            st.markdown("---")

        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.markdown("### 📝 Detailed View: Raw Transcript")
            # If we didn't use comparison mode, show transcript here
            if not st.session_state.original_dialogue:
                st.text_area("Diarized Transcription", value=st.session_state.transcript, height=400)
            else:
                st.caption("See comparison above for conversational detail.")
                st.text_area("Full Single-Block Transcript", value=st.session_state.raw_transcript, height=200)
            
        with res_col2:
            st.markdown("### 📋 Clinical Summary")
            st.text_area("AI Generated Note", value=st.session_state.summary, height=400)
            
            if st.session_state.processing_complete:
                st.download_button(
                    label="Download Clinical Note",
                    data=st.session_state.summary,
                    file_name="consult_note.txt",
                    mime="text/plain"
                )

elif page == "📂 History":
    st.markdown("## 📂 Recording History")
    st.info("Feature coming soon: View and search past dictations.")

elif page == "⚙️ Settings":
    st.markdown("## ⚙️ Settings")
    st.text_input("Doctor Name", value="Dr. Default")
    st.selectbox("Summary Style", ["Narrative", "SOAP", "Procedure Note"])

