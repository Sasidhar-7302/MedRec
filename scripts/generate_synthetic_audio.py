"""
Convert synthetic GI dialogues into <=30s WAV chunks using pyttsx3.

Usage:
    python scripts/generate_synthetic_audio.py --limit 200 --clear-output
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, List

import pyttsx3

SPEAKER_ALIASES = {"doctor", "patient", "nurse", "assistant"}


def normalize_lines(dialogue: str) -> List[str]:
    cleaned: List[str] = []
    for line in dialogue.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            speaker, content = line.split(":", 1)
            if speaker.lower() in SPEAKER_ALIASES:
                line = content.strip()
        cleaned.append(line)
    return cleaned


def chunk_lines(lines: List[str], max_chars: int, max_lines: int) -> Iterable[str]:
    chunk: List[str] = []
    char_count = 0
    for line in lines:
        next_len = len(line) + 1
        if chunk and (char_count + next_len > max_chars or len(chunk) >= max_lines):
            yield " ".join(chunk)
            chunk = []
            char_count = 0
        chunk.append(line)
        char_count += next_len
    if chunk:
        yield " ".join(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic GI WAV clips.")
    parser.add_argument("--jsonl", type=Path, default=Path("data/synthetic_gi_pairs.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_audio"))
    parser.add_argument("--segments-jsonl", type=Path, default=Path("data/synthetic_segments.jsonl"))
    parser.add_argument("--limit", type=int, default=500, help="Number of dialogues to convert (max).")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=220,
        help="Approximate maximum characters per chunk (keeps audio under 30s).",
    )
    parser.add_argument("--max-lines", type=int, default=3, help="Max dialogue lines per chunk.")
    parser.add_argument("--rate", type=int, default=165, help="TTS rate (words per minute).")
    parser.add_argument("--clear-output", action="store_true", help="Delete existing WAV files before generating.")
    args = parser.parse_args()

    entries = [
        json.loads(line)
        for line in args.jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if args.clear_output and args.output.exists():
        shutil.rmtree(args.output)

    args.output.mkdir(parents=True, exist_ok=True)

    engine = pyttsx3.init()
    engine.setProperty("rate", args.rate)

    manifest_records: List[dict] = []
    dialogue_count = 0
    audio_count = 0

    for idx, entry in enumerate(entries):
        if dialogue_count >= args.limit:
            break

        lines = normalize_lines(entry["dialogue"])
        if not lines:
            continue

        chunk_id = 0
        for chunk_text in chunk_lines(lines, args.max_chars, args.max_lines):
            if not chunk_text.strip():
                continue
            filename = f"dialogue_{idx:04d}_seg_{chunk_id:02d}.wav"
            wav_path = args.output / filename
            engine.save_to_file(chunk_text, str(wav_path))
            engine.runAndWait()
            manifest_records.append(
                {
                    "audio": str(wav_path.resolve()),
                    "text": chunk_text,
                    "dialogue_id": idx,
                    "segment_id": chunk_id,
                }
            )
            chunk_id += 1
            audio_count += 1

        dialogue_count += 1

    if manifest_records:
        args.segments_jsonl.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in manifest_records),
            encoding="utf-8",
        )

    print(f"Generated {audio_count} clips from {dialogue_count} dialogues into {args.output}")
    print(f"Segment manifest: {args.segments_jsonl}")


if __name__ == "__main__":
    main()
