"""Quick test of transcription and summarization."""

import os
# Fix OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from app.transcriber import WhisperTranscriber
from app.summarizer import OllamaSummarizer
from app.config import load_config
from app.terminology import apply_corrections
from pathlib import Path
import time

config = load_config()

# Test transcription
print("Testing transcription...")
transcriber = WhisperTranscriber(config.whisper)
print(f"Using engine: {transcriber._engine}")

# Find a sample audio file
sessions_dir = Path("local_storage/sessions")
sample_audio = None
if sessions_dir.exists():
    for session_dir in sessions_dir.glob("session_*"):
        audio_files = list(session_dir.glob("*.wav"))
        if audio_files:
            sample_audio = audio_files[0]
            break

if sample_audio:
    print(f"Transcribing: {sample_audio.name}")
    start = time.perf_counter()
    result = transcriber.transcribe(sample_audio)
    runtime = time.perf_counter() - start
    cleaned = apply_corrections(result.text)
    
    print(f"\nTranscription Results:")
    print(f"Runtime: {runtime:.2f}s")
    print(f"Length: {len(cleaned)} chars")
    print(f"Speed: {len(cleaned)/runtime:.1f} chars/sec")
    print(f"\nText: {cleaned[:200]}...")
else:
    print("No sample audio found")

# Test summarization
print("\n\nTesting summarization...")
summarizer = OllamaSummarizer(config.summarizer)

if summarizer.health_check():
    print("Ollama is running")
    test_text = "Patient is a 50-year-old male with pancolitis. Currently reporting 3-4 bowel movements per day with mild symptoms and normal eating. No rash, abdominal pain or weight loss. Last colonoscopy showed mild inflammation in sigmoid colon and random biopsies negative for dysplasia."
    
    print("Generating summary...")
    start = time.perf_counter()
    try:
        result = summarizer.summarize(test_text, style="Narrative")
        runtime = time.perf_counter() - start
        print(f"\nSummary Results:")
        print(f"Runtime: {runtime:.2f}s")
        print(f"Length: {len(result.summary)} chars")
        print(f"\nSummary:\n{result.summary[:500]}...")
    except Exception as e:
        print(f"Error: {e}")
else:
    print("Ollama not responding")

