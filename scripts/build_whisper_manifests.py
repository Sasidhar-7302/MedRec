"""
Build Whisper fine-tuning manifests for GI transcription.

This utility consolidates synthetic GI dialogues (with matching TTS audio)
and any locally captured real encounters into train/validation JSONL files that
Hugging Face `datasets` can consume. Each JSONL row looks like:

    {"audio": "C:/.../dialogue_0001.wav", "text": "clean transcript", "source": "synthetic"}

Run:
    python scripts/build_whisper_manifests.py \
        --synthetic-jsonl data/synthetic_gi_pairs.jsonl \
        --synthetic-audio data/synthetic_audio \
        --real-root data/real_audio \
        --output-dir data/training_manifests
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import soundfile as sf


SpeakerAliases = {"doctor", "patient", "nurse", "assistant"}


@dataclass
class ManifestEntry:
    audio: Path
    text: str
    source: str

    def to_json(self) -> str:
        return json.dumps(
            {"audio": str(self.audio.resolve()), "text": self.text, "source": self.source},
            ensure_ascii=False,
        )


def strip_speaker_labels(dialogue: str) -> str:
    """Mirror the cleaning applied when generating synthetic audio."""
    cleaned: List[str] = []
    for line in dialogue.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            speaker, content = line.split(":", 1)
            if speaker.lower() in SpeakerAliases:
                line = content.strip()
        cleaned.append(line)
    return " ".join(cleaned)


def load_jsonl(path: Path) -> Iterable[dict]:
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        yield json.loads(raw)


def gather_synthetic(
    jsonl_path: Path,
    audio_dir: Path,
    limit: Optional[int],
    segment_manifest: Optional[Path],
    max_duration: float,
) -> List[ManifestEntry]:
    if segment_manifest and segment_manifest.exists():
        return gather_segment_manifest(segment_manifest, limit, max_duration)
    records = list(load_jsonl(jsonl_path))
    entries: List[ManifestEntry] = []
    for idx, record in enumerate(records):
        if limit is not None and idx >= limit:
            break
        wav_path = audio_dir / f"dialogue_{idx:04d}.wav"
        if not wav_path.exists():
            continue
        dialogue = record.get("dialogue") or record.get("text") or ""
        text = strip_speaker_labels(dialogue)
        if not text:
            continue
        if compute_duration_seconds(wav_path) > max_duration:
            continue
        entries.append(ManifestEntry(audio=wav_path, text=text, source="synthetic"))
    return entries


def gather_segment_manifest(
    manifest_path: Path, limit: Optional[int], max_duration: float
) -> List[ManifestEntry]:
    entries: List[ManifestEntry] = []
    for idx, record in enumerate(load_jsonl(manifest_path)):
        if limit is not None and idx >= limit:
            break
        audio_path = Path(record["audio"])
        if not audio_path.exists():
            continue
        duration = compute_duration_seconds(audio_path)
        if duration > max_duration:
            continue
        text = record.get("text", "").strip()
        if not text:
            continue
        entries.append(ManifestEntry(audio=audio_path, text=text, source="synthetic"))
    return entries


def gather_real(root: Path, max_duration: float) -> List[ManifestEntry]:
    entries: List[ManifestEntry] = []
    if not root.exists():
        return entries
    for encounter in sorted(root.glob("encounter_*")):
        audio = encounter / "audio.wav"
        transcript = encounter / "transcript.txt"
        if not audio.exists() or not transcript.exists():
            continue
        if compute_duration_seconds(audio) > max_duration:
            continue
        text = transcript.read_text(encoding="utf-8").strip()
        if not text:
            continue
        entries.append(ManifestEntry(audio=audio, text=text, source="real"))
    return entries


def split_entries(entries: List[ManifestEntry], train_ratio: float, seed: int) -> tuple[list, list]:
    rng = random.Random(seed)
    shuffled = entries.copy()
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def compute_duration_seconds(path: Path) -> float:
    with sf.SoundFile(str(path)) as f:
        return len(f) / float(f.samplerate)


def write_manifest(path: Path, entries: List[ManifestEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(entry.to_json())
            fh.write("\n")


def build_inventory(entries: List[ManifestEntry]) -> dict:
    total_seconds = sum(compute_duration_seconds(e.audio) for e in entries)
    return {"count": len(entries), "hours": round(total_seconds / 3600.0, 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Whisper JSONL manifests for GI ASR training.")
    parser.add_argument("--synthetic-jsonl", type=Path, default=Path("data/synthetic_gi_pairs.jsonl"))
    parser.add_argument("--synthetic-audio", type=Path, default=Path("data/synthetic_audio"))
    parser.add_argument("--synthetic-manifest", type=Path, default=Path("data/synthetic_segments.jsonl"))
    parser.add_argument("--synthetic-limit", type=int, default=None, help="Optional cap on synthetic samples.")
    parser.add_argument("--real-root", type=Path, default=Path("data/real_audio"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/training_manifests"))
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--inventory-json", type=Path, default=Path("reports/data_inventory.json"))
    parser.add_argument("--max-duration", type=float, default=30.0, help="Maximum clip length (seconds).")
    args = parser.parse_args()

    synthetic_entries = gather_synthetic(
        args.synthetic_jsonl,
        args.synthetic_audio,
        args.synthetic_limit,
        args.synthetic_manifest,
        args.max_duration,
    )
    real_entries = gather_real(args.real_root, args.max_duration)

    combined_entries = synthetic_entries + real_entries
    if not combined_entries:
        raise SystemExit("No audio/text pairs found. Generate synthetic audio or place real encounters first.")

    train_entries, val_entries = split_entries(combined_entries, args.train_ratio, args.seed)
    if not val_entries:
        raise SystemExit("Validation split is empty. Reduce --train-ratio or add more data.")

    prefix = "whisper_gi"
    write_manifest(args.output_dir / f"{prefix}_train.jsonl", train_entries)
    write_manifest(args.output_dir / f"{prefix}_val.jsonl", val_entries)

    # Optional: also expose smaller diagnostic subsets (tiny) for quick smoke tests.
    tiny_train = train_entries[: min(32, len(train_entries))]
    tiny_val = val_entries[: min(8, len(val_entries))]
    write_manifest(args.output_dir / f"{prefix}_tiny_train.jsonl", tiny_train)
    write_manifest(args.output_dir / f"{prefix}_tiny_val.jsonl", tiny_val)

    inventory = {
        "synthetic": build_inventory(synthetic_entries),
        "real": build_inventory(real_entries) if real_entries else {"count": 0, "hours": 0.0},
        "train": build_inventory(train_entries),
        "val": build_inventory(val_entries),
    }
    args.inventory_json.parent.mkdir(parents=True, exist_ok=True)
    args.inventory_json.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(f"Wrote manifests to {args.output_dir} and inventory to {args.inventory_json}")


if __name__ == "__main__":
    main()
