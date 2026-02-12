import sys
from pathlib import Path
from app.config import AppConfig
from app.transcriber import WhisperTranscriber
from app.summarizer import OllamaSummarizer

def main():
    print("Starting End-to-End Verification...")
    
    # 1. Setup
    config = AppConfig.load(Path("config.json"))
    transcriber = WhisperTranscriber(config.whisper)
    summarizer = OllamaSummarizer(config.summarizer)
    
    audio_path = Path("sample.wav")
    if not audio_path.exists():
        print("Error: sample.wav not found!")
        sys.exit(1)
        
    # 2. Transcribe
    print(f"Transcribing {audio_path}...")
    try:
        result = transcriber.transcribe(audio_path)
        print(f"Transcription successful. Text length: {len(result.text)}")
        print(f"Text: {result.text}")
    except Exception as e:
        print(f"Transcription failed: {e}")
        sys.exit(1)

    # 3. Summarize (even if text is garbage/silence, just to test the pipeline)
    print("Summarizing...")
    try:
        # If transcription is empty (sine wave might be silence/hallucination), use dummy text for summary test
        text_to_summarize = result.text if len(result.text) > 5 else "Patient is a 45 year old male presenting with abdominal pain for 2 weeks."
        print(f"Input text for summary: {text_to_summarize}")
        
        summary = summarizer.summarize(text_to_summarize)
        print("Summarization successful.")
        print(f"Summary:\n{summary}")
    except Exception as e:
        print(f"Summarization failed: {e}")
        sys.exit(1)

    print("Verification Complete: SUCCESS")

if __name__ == "__main__":
    main()
