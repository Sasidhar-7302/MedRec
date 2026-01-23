"""Record a fixed-duration WAV file for quick tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import sounddevice as sd
import soundfile as sf


def record(duration: float, output: Path) -> None:
    sample_rate = 16000
    channels = 1
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Recording {duration}s to {output}")
    data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype="int16")
    sd.wait()
    sf.write(output, data, sample_rate)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, help="Output WAV path")
    parser.add_argument("--duration", type=float, default=5.0)
    args = parser.parse_args()
    record(args.duration, args.output)
