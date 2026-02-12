
import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.config import AppConfig
from app.two_pass_summarizer import TwoPassSummarizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("verify_e2e")

def run_sample_summary(filename="GAS0007_transcription.txt"):
    base_dir = Path(__file__).parent
    audio_dir = base_dir / "data" / "GiAudiotest"
    transcript_path = audio_dir / filename
    config_path = base_dir / "config.json"

    if not transcript_path.exists():
        logger.error(f"Transcript file not found: {transcript_path}")
        return

    logger.info("Loading configuration...")
    try:
        app_config = AppConfig.load(config_path)
        summarizer = TwoPassSummarizer(app_config.summarizer)
    except Exception as e:
        logger.error(f"Failed to initialize summarizer: {e}")
        return

    logger.info(f"Reading transcript from {filename}...")
    with open(transcript_path, "r", encoding="utf-8") as f:
        dialogue = f.read()

    logger.info("Running summarization...")
    try:
        result = summarizer.summarize(dialogue)
        print("\n" + "="*60)
        print("GENERATED CLINICAL NOTE")
        print("="*60)
        
        # Format the StructuredSummary object
        summary_text = f"HPI: {result.hpi}\n\n"
        
        if result.findings:
            summary_text += "Findings:\n" + "\n".join([f"- {f}" for f in result.findings]) + "\n\n"
            
        if result.assessment:
            summary_text += "Assessment:\n" + "\n".join([f"{i+1}. {a}" for i, a in enumerate(result.assessment)]) + "\n\n"
            
        if result.plan:
            summary_text += "Plan:\n" + "\n".join([f"- {p}" for p in result.plan]) + "\n\n"
            
        if result.medications:
            summary_text += "Medications/Orders:\n" + "\n".join([f"- {m}" for m in result.medications]) + "\n\n"
            
        summary_text += f"Follow-up: {result.followup}"
        
        print(summary_text)
        print("="*60)
        print("DEBUG: Raw Extraction")
        print(result.raw_extraction)
        print("="*60)
        print(f"Runtime: {result.runtime_s:.2f}s")
    except Exception as e:
        logger.error(f"Summarization failed: {e}")

if __name__ == "__main__":
    run_sample_summary()
