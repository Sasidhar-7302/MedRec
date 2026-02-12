
import logging
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import torch first to ensure it loads its own bundled cuDNN/CUDA libs
# This prevents conflict with 'cuda_libs' potentially added to PATH by transcriber.py
import torch

from app.config import load_config
from app.transcriber import WhisperTranscriber
from app.two_pass_summarizer import TwoPassSummarizer

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def main():
    setup_logging()
    logger = logging.getLogger("verify_giear")

    logger.info("Loading config...")
    config = load_config()
    
    # Force enable diarization for this test if not already
    if not config.whisper.diarization.enabled:
        logger.info("Forcing diarization enabled for test...")
        config.whisper.diarization.enabled = True
        config.whisper.diarization.provider = "whisperx"

    audio_path = Path("data/GiAudiotest/GAS0001.mp3")
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        return

    logger.info("Initializing Transcriber...")
    transcriber = WhisperTranscriber(config.whisper)
    
    logger.info(f"Transcribing {audio_path}...")
    result = transcriber.transcribe(audio_path, progress_cb=lambda msg: print(f"  -> {msg}"))
    
    logger.info("Transcription complete!")
    print("\n--- Raw Diarized Transcript ---")
    print(result.text[:500] + "..." if len(result.text) > 500 else result.text)
    print("-------------------------------\n")
    
    if len(result.text) > 1000:
        mid_start = len(result.text) // 2
        print("\n--- Raw Diarized Transcript (Middle Snippet) ---")
        print(result.text[mid_start:mid_start+500] + "...")
        print("------------------------------------------------\n")

    logger.info("Initializing Summarizer...")
    summarizer = TwoPassSummarizer(config.summarizer)
    
    # Test Diarize/Mapping separately first
    logger.info("Testing Speaker Mapping (Diarize stage)...")
    mapped_transcript = summarizer.diarize(result.text)
    
    print("\n--- Mapped Transcript (First 500 chars) ---")
    print(mapped_transcript[:500] + "...")
    print("-------------------------------------------\n")

    logger.info("Generating Summary...")
    summary = summarizer.summarize_text(mapped_transcript)
    
    print("\n=== FINAL CLINICAL NOTE ===")
    print(summary)
    print("===========================")

if __name__ == "__main__":
    main()
