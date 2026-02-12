
import os
import re
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.config import AppConfig
from app.two_pass_summarizer import TwoPassSummarizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("batch_summarize")

def extract_hpi_fallback(raw_text: str) -> str:
    """Fallback to extract HPI from raw extraction if structuring failed."""
    if not raw_text:
        return ""
    
    # Look for Patient History section
    patterns = [
        r"PATIENT HISTORY:?\s*\n(.*?)(?=\n\d+\.|\n[A-Z]+|$)",
        r"1\.\s*PATIENT HISTORY:?\s*\n(.*?)(?=\n\d+\.|\n[A-Z]+|$)",
        r"History:?\s*\n(.*?)(?=\n\d+\.|\n[A-Z]+|$)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""

def format_note(result) -> str:
    """Format the StructuredSummary object into a text note."""
    hpi = result.hpi
    if not hpi or len(hpi) < 10:
        logger.warning("HPI missing in structured output, attempting fallback...")
        fallback = extract_hpi_fallback(result.raw_extraction)
        if fallback:
            hpi = fallback
            logger.info("Fallback HPI extracted.")

    summary_text = f"HPI (History of Present Illness):\n{hpi}\n\n"
    
    if result.findings:
        summary_text += "Findings:\n" + "\n".join([f"- {f}" for f in result.findings]) + "\n\n"
        
    if result.assessment:
        summary_text += "Assessment:\n" + "\n".join([f"{i+1}. {a}" for i, a in enumerate(result.assessment)]) + "\n\n"
        
    if result.plan:
        summary_text += "Plan:\n" + "\n".join([f"- {p}" for p in result.plan]) + "\n\n"
        
    if result.medications:
        summary_text += "Medications/Orders:\n" + "\n".join([f"- {m}" for m in result.medications]) + "\n\n"
        
    summary_text += f"Follow-up:\n{result.followup}"
    
    return summary_text

def run_batch_summarization():
    base_dir = Path(__file__).parent
    audio_dir = base_dir / "data" / "GiAudiotest"
    config_path = base_dir / "config.json"

    if not audio_dir.exists():
        logger.error(f"Directory not found: {audio_dir}")
        return

    logger.info("Loading configuration...")
    try:
        app_config = AppConfig.load(config_path)
        summarizer = TwoPassSummarizer(app_config.summarizer)
    except Exception as e:
        logger.error(f"Failed to initialize summarizer: {e}")
        return

    # Find transcription files
    trans_files = list(audio_dir.glob("*_transcription.txt"))
    if not trans_files:
        logger.warning(f"No transcription files found in {audio_dir}.")
        return

    logger.info(f"Found {len(trans_files)} transcription files.")
    
    print("\n" + "="*60)
    print(f"{'File':<30} | {'Status':<10} | {'Runtime':<10}")
    print("-" * 60)

    for trans_file in trans_files:
        # if "GAS0005" not in trans_file.name: continue
        logger.info(f"Processing {trans_file.name}...")
        
        try:
            with open(trans_file, "r", encoding="utf-8") as f:
                dialogue = f.read()
            
            result = summarizer.summarize(dialogue)
            note_content = format_note(result)
            
            # Save clinical note
            note_path = audio_dir / f"{trans_file.stem.replace('_transcription', '')}_clinical_note.txt"
            with open(note_path, "w", encoding="utf-8") as f:
                f.write(note_content)
                
            print(f"{trans_file.name:<30} | {'DONE':<10} | {result.runtime_s:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to summarize {trans_file.name}: {e}")
            print(f"{trans_file.name:<30} | {'ERROR':<10} | {'-':<10}")

    print("="*60)
    print("Batch processing complete.")

if __name__ == "__main__":
    run_batch_summarization()
