import sys
import os
import json
import logging
import jiwer
import re
from pathlib import Path
from dataclasses import dataclass

# Add project root to path
sys.path.append(os.getcwd())

# Dummy Config class since importing from app.config might be tricky if paths vary
@dataclass
class SummarizerConfig:
    provider: str
    model: str
    base_url: str
    timeout_s: int = 600
    max_tokens: int = 4096
    context_window: int = 8192
    use_self_correction: bool = True

from app.transcript_polisher import TranscriptPolisher
# Configure logging
logging.basicConfig(level=logging.INFO)

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
    ref_clean = clean(reference)
    hyp_clean = clean(hypothesis)
    if not ref_clean: return 0.0
    return jiwer.wer(ref_clean, hyp_clean)

def main():
    # Target Case: GAS0005 (Pediatric Case)
    case_id = "GAS0005"
    base_dir = Path("c:/Users/yepur/Desktop/My_Projects/GI_Scribe")
    
    # 1. Load Raw Transcript (Hypothesis)
    trans_path = base_dir / f"data/GiAudiotest/results/{case_id}_transcription.txt"
    if not trans_path.exists():
        print(f"Error: {trans_path} not found.")
        return
    with open(trans_path, "r", encoding="utf-8") as f:
        raw_transcript = f.read()

    # 2. Load Ground Truth (Reference)
    gt_path = base_dir / f"data/GiTestValid/{case_id}.txt"
    if not gt_path.exists():
        print(f"Error: Ground truth {gt_path} not found.")
        return
    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = f.read()

    # 3. Calculate Baseline WER
    baseline_wer = calculate_wer(ground_truth, raw_transcript)
    print(f"[{case_id}] Baseline WER: {baseline_wer:.4f} (Accuracy: {1-baseline_wer:.2%})")

    # 4. Polish Transcript
    model_name = "medllama2" # Default
    base_url = "http://localhost:11434"
    config_path = base_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                conf = json.load(f)
                # Correctly access nested config
                summ_conf = conf.get("summarizer", {})
                model_name = summ_conf.get("model", model_name)
                base_url = summ_conf.get("base_url", base_url)
        except Exception as e:
            print(f"Config load error: {e}")

    config = SummarizerConfig(
        provider="ollama",
        model=model_name,
        base_url=base_url
    )
    
    print(f"Polishing transcript using {model_name}...")
    polisher = TranscriptPolisher(config)
    result = polisher.polish(raw_transcript)
    
    # Save polished transcript
    out_path = base_dir / f"data/GiAudiotest/results/{case_id}_polished.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result.polished_text)
    print(f"Polished transcript saved to {out_path}")

    # 5. Calculate Polished WER
    polished_wer = calculate_wer(ground_truth, result.polished_text)
    print(f"[{case_id}] Polished WER: {polished_wer:.4f} (Accuracy: {1-polished_wer:.2%})")
    
    improvement = baseline_wer - polished_wer
    print(f"Improvement: +{improvement*100:.2f}%")

if __name__ == "__main__":
    main()
