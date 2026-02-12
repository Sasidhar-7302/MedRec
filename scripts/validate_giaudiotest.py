import os
import json
import time
import logging
import re
from pathlib import Path
import jiwer
import torch

from app.config import load_config
from app.transcriber import WhisperTranscriber
from app.two_pass_summarizer import TwoPassSummarizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("giaudio_validator")

# Define paths
BASE_DIR = Path("data/GiAudiotest")
GT_DIR = Path("data/GiTestValid")
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate (WER)."""
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

    ref_clean = clean(reference)
    hyp_clean = clean(hypothesis)
    
    if not ref_clean: return 0.0
    return jiwer.wer(ref_clean, hyp_clean)

def validate_giaudio():
    """Run MedRec over the GiAudiotest folder."""
    logger.info("Loading config and initializing models...")
    config = load_config()
    
    # Ensure diarization is enabled
    config.whisper.diarization.enabled = True
    config.whisper.diarization.provider = "whisperx"
    
    transcriber = WhisperTranscriber(config.whisper)
    summarizer = TwoPassSummarizer(config.summarizer)
    
    # Specific files requested
    target_files = ["GAS0001.mp3", "GAS0002.mp3", "GAS0003.mp3", "GAS0004.mp3", "GAS0005.mp3", "GAS0007.mp3"]
    audio_files = [BASE_DIR / f for f in target_files]
    
    results = []
    
    for audio_path in audio_files:
        if not audio_path.exists():
            logger.warning(f"File not found: {audio_path}")
            continue
            
        case_id = audio_path.stem
        logger.info(f"Processing {case_id}...")
        
        # Load ground truth transcription from GiTestValid
        gt_trans_path = GT_DIR / f"{case_id}.txt"
        gt_text = ""
        if gt_trans_path.exists():
            with open(gt_trans_path, "r", encoding="utf-8") as f:
                gt_text = f.read()
        else:
            logger.warning(f"Ground truth transcription not found in GiTestValid for {case_id}")

        start_time = time.perf_counter()
        try:
            # 0. Get Audio Duration
            import subprocess
            def get_duration(p):
                cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(p)]
                try:
                    return float(subprocess.check_output(cmd).decode().strip())
                except: return 0.0
            
            audio_duration = get_duration(audio_path)

            # 1. Transcribe
            t_start = time.perf_counter()
            trans_result = transcriber.transcribe(audio_path)
            trans_time = time.perf_counter() - t_start
            
            # 2. Diarize/Map & Summarize
            s_start = time.perf_counter()
            mapped_transcript = summarizer.diarize(trans_result.text)
            summary = summarizer.summarize_text(mapped_transcript)
            summ_time = time.perf_counter() - s_start
            
            total_time = time.perf_counter() - start_time
            
            # 3. Calculate WER (if GT exists)
            wer = calculate_wer(gt_text, trans_result.text) if gt_text else None
            
            case_result = {
                "case_id": case_id,
                "audio_duration_s": audio_duration,
                "trans_time_s": trans_time,
                "summ_time_s": summ_time,
                "total_time_s": total_time,
                "wer": wer,
                "accuracy": (1.0 - wer) if wer is not None else None,
                "summary": summary
            }
            results.append(case_result)
            
            acc_str = f"{1-wer:.2%}" if wer is not None else "N/A"
            logger.info(f"Done {case_id} | Acc: {acc_str} | Audio: {audio_duration:.1f}s | Trans: {trans_time:.1f}s | Summ: {summ_time:.1f}s")
             
            # Save individual result
            with open(RESULTS_DIR / f"{case_id}_result.json", "w", encoding="utf-8") as f:
                json.dump(case_result, f, indent=2)
                
            # Save generated summary to compare with clinical note
            # Save generated summary to compare with clinical note
            with open(RESULTS_DIR / f"{case_id}_generated_note.txt", "w", encoding="utf-8") as f:
                f.write(summary)

            # Save raw transcription for debugging
            with open(RESULTS_DIR / f"{case_id}_transcription.txt", "w", encoding="utf-8") as f:
                f.write(trans_result.text)

        except Exception as e:
            logger.error(f"Failed to process {case_id}: {e}")
            results.append({"case_id": case_id, "error": str(e)})

    # Final Report
    if not results: return
    
    valid_results = [r for r in results if r.get("wer") is not None]
    
    print("\n" + "="*40)
    print("GIAUDIOTEST BENCHMARK COMPLETE")
    print(f"Total Cases: {len(results)}")
    
    if valid_results:
        avg_wer = sum(r["wer"] for r in valid_results) / len(valid_results)
        avg_acc = 1.0 - avg_wer
        avg_trans = sum(r.get("trans_time_s", 0) for r in valid_results) / len(valid_results)
        avg_summ = sum(r.get("summ_time_s", 0) for r in valid_results) / len(valid_results)
        avg_audio = sum(r.get("audio_duration_s", 0) for r in valid_results) / len(valid_results)
        
        print(f"Avg Accuracy: {avg_acc:.2%}")
        print(f"Avg Audio Len: {avg_audio:.1f}s")
        print(f"Avg Transcribe: {avg_trans:.1f}s")
        print(f"Avg Summarize: {avg_summ:.1f}s")
    else:
        print("No WER metrics available (missing ground truth?)")
        
    print("="*40)
    
    # Save aggregate report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results
    }
    with open(RESULTS_DIR / "final_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    validate_giaudio()
