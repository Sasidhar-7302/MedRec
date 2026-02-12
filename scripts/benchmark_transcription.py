import time
import os
from pathlib import Path
from faster_whisper import WhisperModel

# Inject CUDA Path
CUDA_LIBS = Path(r"c:\Users\yepur\Desktop\My_Projects\GI_Scribe\cuda_libs")
if CUDA_LIBS.exists():
    os.environ["PATH"] = str(CUDA_LIBS) + os.pathsep + os.environ.get("PATH", "")

def benchmark():
    model_size = "small.en"
    device = "cuda" # Current config
    # device = "cpu" # Fallback
    
    print(f"Initializing Whisper model ({model_size}) on {device}...")
    start_load = time.time()
    try:
        model = WhisperModel(model_size, device=device, compute_type="int8")
        print(f"Model loaded in {time.time() - start_load:.2f}s")
    except Exception as e:
        print(f"CUDA initialization failed: {e}. Falling back to CPU.")
        start_load = time.time()
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print(f"Model loaded (CPU) in {time.time() - start_load:.2f}s")

    audio_path = "kaggle_dialogue.mp3"
    if not os.path.exists(audio_path):
        print(f"Error: {audio_path} not found.")
        return

    print(f"Transcribing {audio_path}...")
    start_transcribe = time.time()
    segments, info = model.transcribe(audio_path, beam_size=5)
    
    text = ""
    for segment in segments:
        text += segment.text + " "
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
    
    end_time = time.time()
    print("-" * 20)
    print(f"Transcription finished in {end_time - start_transcribe:.2f}s")
    print(f"Audio duration: {info.duration:.2f}s")
    print(f"Real-time factor: {info.duration / (end_time - start_transcribe):.2f}x")
    print("-" * 20)

if __name__ == "__main__":
    benchmark()
