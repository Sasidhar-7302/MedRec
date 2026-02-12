
import os
import re
import sys
import logging
from pathlib import Path
from typing import List, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.config import AppConfig
from app.transcriber import WhisperTranscriber
from app.gi_post_processor import GIPostProcessor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("test_whisper")

def normalize_text(text: str) -> str:
    """Normalize text for comparison (lower case, remove punctuation)."""
    # Remove speaker labels if present (D: / P:)
    text = re.sub(r"^[DP]:\s*", "", text, flags=re.MULTILINE)
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings (word level)."""
    words1 = s1.split()
    words2 = s2.split()
    
    if len(words1) < len(words2):
        return levenshtein_distance(s2, s1)

    if len(words2) == 0:
        return len(words1)

    previous_row = range(len(words2) + 1)
    for i, c1 in enumerate(words1):
        current_row = [i + 1]
        for j, c2 in enumerate(words2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate."""
    if not reference:
        return 1.0 if hypothesis else 0.0
        
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)
    
    distance = levenshtein_distance(ref_norm, hyp_norm)
    ref_len = len(ref_norm.split())
    
    if ref_len == 0:
        return 1.0 if hyp_norm else 0.0
        
    return distance / ref_len

def run_accuracy_test():
    # Paths
    base_dir = Path(__file__).parent
    audio_dir = base_dir / "data" / "GiAudiotest"
    valid_dir = base_dir / "data" / "GiTestValid"
    config_path = base_dir / "config.json"

    if not audio_dir.exists():
        logger.error(f"Audio directory not found: {audio_dir}")
        return
    if not valid_dir.exists():
        logger.error(f"Validation directory not found: {valid_dir}")
        return

    # Load config and init transcriber
    logger.info("Loading configuration...")
    try:
        app_config = AppConfig.load(config_path)
        transcriber = WhisperTranscriber(app_config.whisper)
    except Exception as e:
        logger.error(f"Failed to initialize transcriber: {e}")
        return

    # Find audio files
    audio_files = list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.m4a"))
    if not audio_files:
        logger.warning(f"No audio files found in {audio_dir}. Please place MP3/WAV files there.")
        return

    logger.info(f"Found {len(audio_files)} audio files.")
    
    total_wer = 0.0
    passed_files = 0

    print("\n" + "="*60)
    print(f"{'File':<30} | {'WER':<10} | {'Status':<10}")
    print("-" * 60)

    for audio_file in audio_files:
        # Match validation file
        # Check standard matching: name.mp3 -> name.txt
        valid_file = valid_dir / f"{audio_file.stem}.txt"
        
        if not valid_file.exists():
            logger.warning(f"No validation file found for {audio_file.name} (checked {valid_file.name})")
            continue

        # Valid text
        try:
            with open(valid_file, "r", encoding="utf-8") as f:
                reference_text = f.read()
        except Exception as e:
            logger.error(f"Error reading {valid_file.name}: {e}")
            continue

        # Transcribe
        logger.info(f"Transcribing {audio_file.name}...")
        try:
            result = transcriber.transcribe(audio_file)
            hypothesis_text = result.text
        except Exception as e:
            logger.error(f"Transcription failed for {audio_file.name}: {e}")
            print(f"{audio_file.name:<30} | {'ERROR':<10} | {'FAIL':<10}")
            continue

        # Calculate WER
        wer = calculate_wer(reference_text, hypothesis_text)
        total_wer += wer
        passed_files += 1
        
        status = "EXCELLENT" if wer < 0.1 else "GOOD" if wer < 0.2 else "FAIR" if wer < 0.3 else "POOR"
        print(f"{audio_file.name:<30} | {wer:.2%}    | {status:<10}")
        
        # Save output for review
        out_path = audio_dir / f"{audio_file.stem}_transcription.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(hypothesis_text)

    if passed_files > 0:
        avg_wer = total_wer / passed_files
        print("="*60)
        print(f"Average WER: {avg_wer:.2%}")
        print(f"Accuracy: {1.0 - avg_wer:.2%}")
        print("="*60)
    else:
        print("\nNo valid comparisons performed.")

if __name__ == "__main__":
    run_accuracy_test()
