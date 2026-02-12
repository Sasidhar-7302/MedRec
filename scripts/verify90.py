from app.config import AppConfig
from app.two_pass_summarizer import TwoPassSummarizer
from pathlib import Path
import logging

# Configure logging to see the stages
logging.basicConfig(level=logging.INFO)

def verify():
    print("Loading config...")
    config = AppConfig.load(Path("config.json"))
    summarizer = TwoPassSummarizer(config.summarizer)

    # Read the transcript for GAS0005 (which had hallucinations before)
    transcript_path = Path("data/GiAudiotest/GAS0005_transcription.txt")
    if not transcript_path.exists():
        print(f"File not found: {transcript_path}")
        return

    transcript = transcript_path.read_text(encoding="utf-8")
    
    print(f"Processing {transcript_path.name} with 4-stage pipeline...")
    result = summarizer.summarize(transcript)
    
    print("\n--- FINAL STRUCTURED SUMMARY ---")
    print(f"HPI: {result.hpi}")
    print(f"Findings: {result.findings}")
    print(f"Assessment: {result.assessment}")
    print(f"Plan: {result.plan}")
    print(f"Medications: {result.medications}")
    print(f"Follow-up: {result.followup}")
    
    # Save to a new file for manual inspection
    with open("verify90_output.txt", "w", encoding="utf-8") as f:
        f.write(f"HPI: {result.hpi}\n\n")
        f.write(f"Findings: {result.findings}\n\n")
        f.write(f"Assessment: {result.assessment}\n\n")
        f.write(f"Plan: {result.plan}\n\n")
        f.write(f"Medications: {result.medications}\n\n")
        f.write(f"Follow-up: {result.followup}\n\n")
    
    print("\nResults saved to verify90_output.txt")

if __name__ == "__main__":
    verify()
