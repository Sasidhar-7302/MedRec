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
logger = logging.getLogger("batch_validator")

AUDIO_DIR = Path("data/synthetic/audio")
TRANSCRIPT_DIR = Path("data/synthetic/transcripts")
RESULTS_DIR = Path("data/synthetic/results")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate (WER)."""
    def clean(text):
        # Remove timestamps
        text = re.sub(r'\[\d{2}:\d{2}.*?\]', '', text)
        # Remove speaker labels
        text = re.sub(r'Doctor:|Patient:|Speaker \d+:', '', text, flags=re.IGNORECASE)
        # Normalize whitespace and lower case
        text = re.sub(r'\s+', ' ', text).strip().lower()
        # Remove punctuation for core text comparison
        text = re.sub(r'[^\w\s]', '', text)
        return text

    ref_clean = clean(reference)
    hyp_clean = clean(hypothesis)
    
    if not ref_clean: return 0.0
    return jiwer.wer(ref_clean, hyp_clean)

def validate_batch(num_files=None):
    """Run MedRec over the synthetic batch and report metrics."""
    logger.info("Loading config and initializing models...")
    config = load_config()
    
    # Ensure diarization is enabled for validation
    config.whisper.diarization.enabled = True
    config.whisper.diarization.provider = "whisperx"
    
    transcriber = WhisperTranscriber(config.whisper)
    summarizer = TwoPassSummarizer(config.summarizer)
    
    audio_files = sorted(list(AUDIO_DIR.glob("*.mp3")))
    if num_files:
        audio_files = audio_files[:num_files]
    
    if not audio_files:
        logger.error("No audio files found for validation.")
        return

    results = []
    
    for audio_path in audio_files:
        case_id = audio_path.stem
        logger.info(f"Processing {case_id}...")
        
        # Load ground truth
        gt_path = TRANSCRIPT_DIR / f"{case_id}.txt"
        if not gt_path.exists():
            logger.warning(f"Ground truth not found for {case_id}, skipping.")
            continue
            
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_text = f.read()
            
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
            
            # 4. Calculate WER
            wer = calculate_wer(gt_text, trans_result.text)
            
            # 5. Validate Summary structure
            sections_found = {
                "HPI": "HPI" in summary,
                "Findings": "Findings" in summary,
                "Assessment": "Assessment" in summary,
                "Plan": "Plan" in summary
            }
            
            case_result = {
                "case_id": case_id,
                "audio_duration_s": audio_duration,
                "trans_time_s": trans_time,
                "summ_time_s": summ_time,
                "total_time_s": total_time,
                "wer": wer,
                "accuracy": 1.0 - wer,
                "sections": sections_found,
                "summary_length": len(summary),
                "summary": summary
            }
            results.append(case_result)
            logger.info(f"Done {case_id} | Acc: {1-wer:.2%} | Audio: {audio_duration:.1f}s | Trans: {trans_time:.1f}s | Summ: {summ_time:.1f}s")
             
            # Save individual result
            with open(RESULTS_DIR / f"{case_id}_result.json", "w", encoding="utf-8") as f:
                json.dump(case_result, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to process {case_id}: {e}")
            results.append({"case_id": case_id, "error": str(e)})

    # Final Report
    if not results: return
    
    valid_results = [r for r in results if r.get("wer") is not None]
    
    if not valid_results:
        logger.warning("No valid results to report.")
        return
        
    avg_wer = sum(r["wer"] for r in valid_results) / len(valid_results)
    avg_acc = 1.0 - avg_wer
    avg_trans = sum(r["trans_time_s"] for r in valid_results) / len(valid_results)
    avg_summ = sum(r["summ_time_s"] for r in valid_results) / len(valid_results)
    avg_audio = sum(r["audio_duration_s"] for r in valid_results) / len(valid_results)
    avg_total = sum(r["total_time_s"] for r in valid_results) / len(valid_results)
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_cases": len(results),
        "successful_cases": len(valid_results),
        "average_wer": avg_wer,
        "average_accuracy": avg_acc,
        "average_audio_duration": avg_audio,
        "average_transcription_time": avg_trans,
        "average_summarization_time": avg_summ,
        "average_total_time": avg_total,
        "results": results
    }
    
    with open("data/synthetic/final_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print("\n" + "="*40)
    print("BATCH VALIDATION COMPLETE")
    print(f"Total Cases: {len(results)}")
    print(f"Avg Accuracy: {avg_acc:.2%}")
    print(f"Avg Audio Len: {avg_audio:.1f}s")
    print(f"Avg Transcribe: {avg_trans:.1f}s")
    print(f"Avg Summarize: {avg_summ:.1f}s")
    print(f"Avg Total RTime: {avg_total:.1f}s")
    print("="*40)

if __name__ == "__main__":
    import sys
    count = int(sys.argv[1]) if len(sys.argv) > 1 else None
    validate_batch(count)
