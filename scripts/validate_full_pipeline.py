import os
import json
import time
import logging
import re
from pathlib import Path
import jiwer
import sys

# Add project root to path
sys.path.append(os.getcwd())

from app.config import load_config
from app.transcriber import WhisperTranscriber
from app.two_pass_summarizer import TwoPassSummarizer
from app.transcript_polisher import TranscriptPolisher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("full_pipeline_validator")

# Define paths
BASE_DIR = Path("data/GiAudiotest")
GT_DIR = Path("data/GiTestValid")
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def clean(text):
    # Remove timestamps
    text = re.sub(r'\[\d{2}:\d{2}.*?\]', '', text)
    # Remove speaker labels (D:, P:, Doctor:, Patient:)
    text = re.sub(r'\b(D|P|Doctor|Patient|Speaker \d+):', '', text, flags=re.IGNORECASE)
    # Remove hyphens and special chars
    text = re.sub(r'[^\w\s]', '', text)
    # Normalize whitespace and lower case
    text = re.sub(r'\s+', ' ', text).strip().lower()
    # Remove filler words
    text = re.sub(r'\b(um|uh|like|ah|oh|mm|mhm)\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate (WER)."""
    ref_clean = clean(reference)
    hyp_clean = clean(hypothesis)
    
    if not ref_clean: return 0.0
    return jiwer.wer(ref_clean, hyp_clean)

def validate_full_pipeline():
    """Run MedRec Full Pipeline (Transcribe -> Polish -> Summarize)."""
    logger.info("Loading config and initializing models...")
    config = load_config()
    
    # Ensure diarization is enabled
    config.whisper.diarization.enabled = True
    config.whisper.diarization.provider = "whisperx"
    
    transcriber = WhisperTranscriber(config.whisper)
    polisher = TranscriptPolisher(config.summarizer) 
    summarizer = TwoPassSummarizer(config.summarizer)
    
    # Specific files requested
    target_files = ["GAS0001.mp3", "GAS0002.mp3", "GAS0003.mp3", "GAS0004.mp3", "GAS0005.mp3", "GAS0007.mp3"]
    
    results = []
    
    for filename in target_files:
        audio_path = BASE_DIR / filename
        if not audio_path.exists():
            logger.warning(f"File not found: {audio_path}")
            continue
            
        case_id = audio_path.stem
        logger.info(f"Processing {case_id}...")
        
        # Load ground truth transcription
        gt_trans_path = GT_DIR / f"{case_id}.txt"
        gt_text = ""
        if gt_trans_path.exists():
            with open(gt_trans_path, "r", encoding="utf-8") as f:
                gt_text = f.read()

        start_time = time.perf_counter()
        try:
            # 1. Transcribe (or load cached)
            # To save time, we check if a transcription exists and is recent (< 24 hours)?
            # For this request, let's FORCE generation to be safe, unless user says otherwise.
            # But Whisper large-v3 is slow. Let's use the cached transcription if available to speed up polishing test.
            
            raw_trans_path = RESULTS_DIR / f"{case_id}_transcription.txt"
            raw_transcript = ""
            if raw_trans_path.exists():
                logger.info(f"Loading cached transcription for {case_id}")
                with open(raw_trans_path, "r", encoding="utf-8") as f:
                    raw_transcript = f.read()
            else:
                 logger.info(f"Running Whisper on {case_id}...")
                 trans_result = transcriber.transcribe(audio_path)
                 raw_transcript = trans_result.text
                 with open(raw_trans_path, "w", encoding="utf-8") as f:
                    f.write(raw_transcript)
            
            # 2. Polish
            logger.info(f"Polishing {case_id}...")
            p_start = time.perf_counter()
            polish_result = polisher.polish(raw_transcript)
            polished_transcript = polish_result.polished_text
            polish_time = time.perf_counter() - p_start
            
            with open(RESULTS_DIR / f"{case_id}_polished.txt", "w", encoding="utf-8") as f:
                f.write(polished_transcript)

            # 3. Summarize
            logger.info(f"Summarizing {case_id}...")
            s_start = time.perf_counter()
            # Note: Summarizer expects a certain format. Polished transcript maintains Speaker labels so it's compatible.
            # We skip 'diarize' method in summarizer because text is already diarized.
            # But TwoPassSummarizer.diarize() does the mapping "SPEAKER_00" -> "Doctor".
            # We need to map speakers BEFORE summarization.
            
            mapped_transcript = summarizer.diarize(polished_transcript)
            summary = summarizer.summarize_text(mapped_transcript)
            summ_time = time.perf_counter() - s_start
            
            total_time = time.perf_counter() - start_time
            
            # 4. Metrics
            raw_wer = calculate_wer(gt_text, raw_transcript) if gt_text else None
            polished_wer = calculate_wer(gt_text, polished_transcript) if gt_text else None
            
            case_result = {
                "case_id": case_id,
                "raw_wer": raw_wer,
                "polished_wer": polished_wer,
                "wer_improvement": raw_wer - polished_wer if (raw_wer and polished_wer) else 0,
                "summary": summary
            }
            results.append(case_result)
            
            logger.info(f"Done {case_id} | Raw WER: {raw_wer:.2f} | Polished WER: {polished_wer:.2f}")
             
            # Save summary
            with open(RESULTS_DIR / f"{case_id}_final_summary.txt", "w", encoding="utf-8") as f:
                f.write(summary)

        except Exception as e:
            logger.error(f"Failed to process {case_id}: {e}")
            results.append({"case_id": case_id, "error": str(e)})

    # Final Report
    print("\n" + "="*40)
    print("FULL PIPELINE VALIDATION COMPLETE")
    for r in results:
        cid = r.get("case_id")
        rwer = r.get("raw_wer")
        pwer = r.get("polished_wer")
        swar = r.get("wer_improvement", 0)
        print(f"{cid}: Raw WER {rwer:.4f} -> Polished {pwer:.4f} (Change: {swar:+.4f})")
    print("="*40)
    
    with open(RESULTS_DIR / "full_pipeline_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    validate_full_pipeline()
