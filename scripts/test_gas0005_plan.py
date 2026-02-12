import sys
import os
import json
import logging
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
    timeout_s: int = 300
    max_tokens: int = 1536
    context_window: int = 4096
    use_self_correction: bool = True

from app.two_pass_summarizer import TwoPassSummarizer

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    # Load the transcription
    base_dir = Path("c:/Users/yepur/Desktop/My_Projects/GI_Scribe")
    trans_path = base_dir / "data/GiAudiotest/results/GAS0005_transcription.txt"
    config_path = base_dir / "config.json"
    
    if not trans_path.exists():
        print(f"Error: {trans_path} not found.")
        return

    with open(trans_path, "r", encoding="utf-8") as f:
        transcript = f.read()

    print(f"Loaded transcript length: {len(transcript)}")

    # Load Config from JSON
    model_name = "medllama2"
    base_url = "http://localhost:11434"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                conf = json.load(f)
                model_name = conf.get("summarizer_model", model_name)
                base_url = conf.get("ollama_url", base_url)
        except Exception as e:
            print(f"Config load error: {e}")

    config = SummarizerConfig(
        provider="ollama",
        model=model_name,
        base_url=base_url
    )
    
    print(f"Using model: {model_name} at {base_url}")
    summarizer = TwoPassSummarizer(config)

    print("Running summarization...")
    summary = summarizer.summarize(transcript)

    print("\n--- GENERATED PLAN ---")
    print(summary.plan) # List[str]
    print("----------------------")

    print("\n--- FULL NOTE ---")
    print(summarizer._format_structured_summary(summary))

if __name__ == "__main__":
    main()
