"""Quick diagnostics for doctors' laptops."""

from __future__ import annotations

import shutil
import sys

import sounddevice as sd

from pathlib import Path

from .config import load_config
from .summarizer import OllamaSummarizer


def main() -> int:
    config = load_config()
    print("== Environment Check ==")

    print("\nAudio devices:")
    devices = [dev for dev in sd.query_devices() if dev["max_input_channels"] > 0]
    for idx, dev in enumerate(devices):
        print(f"  {idx}: {dev['name']} (max {dev['max_input_channels']} ch)")
    if not devices:
        print("  !! No microphone devices detected")

    print("\nwhisper.cpp binary:")
    binary = config.whisper.binary_path
    if shutil.which(binary) or Path(binary).exists():
        print(f"  Found at {binary}")
    else:
        print(f"  !! Missing binary {binary}")
        return 1

    print("\nSummarizer (Ollama) health:")
    summarizer = OllamaSummarizer(config.summarizer)
    if summarizer.health_check():
        print("  Ollama is responding.")
    else:
        print("  !! Ollama endpoint not reachable.")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
