
import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.config import AppConfig
from app.two_pass_summarizer import TwoPassSummarizer, EXTRACTION_PROMPT, STRUCTURING_PROMPT, TEMPLATE_STYLE

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("debug_hpi")

def debug_hpi():
    base_dir = Path(__file__).parent
    audio_dir = base_dir / "data" / "GiAudiotest"
    transcript_path = audio_dir / "GAS0007_transcription.txt"
    config_path = base_dir / "config.json"

    if not transcript_path.exists():
        logger.error(f"File not found: {transcript_path}")
        return

    logger.info("Loading configuration...")
    app_config = AppConfig.load(config_path)
    summarizer = TwoPassSummarizer(app_config.summarizer)

    with open(transcript_path, "r", encoding="utf-8") as f:
        dialogue = f.read()

    logger.info("Running summarization...")
    
    # Manually invoke passes to see intermediate steps
    extraction_prompt = EXTRACTION_PROMPT.format(
        gi_hints=summarizer.gi_hints,
        transcript=dialogue.strip()
    )
    raw_extraction = summarizer._invoke_model(extraction_prompt, temperature=0.1)
    
    print("\n" + "="*30 + " RAW EXTRACTION " + "="*30)
    print(raw_extraction)
    
    structuring_prompt = STRUCTURING_PROMPT.format(
        few_shot_examples=TEMPLATE_STYLE,
        extracted_info=raw_extraction
    )
    raw_structured = summarizer._invoke_model(structuring_prompt, temperature=0.05)
    
    print("\n" + "="*30 + " RAW STRUCTURED " + "="*30)
    print(raw_structured)
    
    # 1. Strip prefix
    stripped = summarizer._strip_conversational_prefix(raw_structured)
    print("\n" + "="*30 + " STRIPPED " + "="*30)
    print(stripped)

    # 2. Add HPI header if missing
    if not stripped.strip().startswith("HPI") and "HPI" not in stripped[:20]:
         stripped = "HPI (History of Present Illness): " + stripped
         print("Prepend HPI header triggered")
    
    # 3. Post-process
    from app.gi_post_processor import process_summary
    processed = process_summary(stripped)
    print("\n" + "="*30 + " PROCESSED " + "="*30)
    print(processed)

    # 4. Enforce structure
    final = summarizer._enforce_structure(processed)
    print("\n" + "="*30 + " FINAL HPI " + "="*30)
    print(f"'{final.hpi}'")

    # Check HPI extraction logic manually on PROCESSED
    hpi_extracted = summarizer._extract_section(processed, "HPI")
    print("\n" + "="*30 + " MANUAL EXTRACT FROM PROCESSED " + "="*30)
    print(f"'{hpi_extracted}'")

    # Check fallback logic
    fallback = "N/A"
    import re
    match = re.search(r"PATIENT HISTORY:?\s*\n(.*?)(?=\n\d+\.|\n[A-Z]+|$)", raw_extraction, re.DOTALL | re.IGNORECASE)
    if match:
        fallback = match.group(1).strip()
    print("\n" + "="*30 + " FALLBACK HPI " + "="*30)
    print(f"'{fallback}'")

if __name__ == "__main__":
    debug_hpi()
