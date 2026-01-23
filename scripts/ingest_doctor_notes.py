"""
Ingest historical doctor notes/summaries into a profile for personalization.

Usage:
    python scripts/ingest_doctor_notes.py --doctor-id dr_smith --name "Dr. Smith" --notes data/notes/dr_smith
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.doctor_profiles import DoctorProfileManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Store historical GI notes for doctor personalization.")
    parser.add_argument("--doctor-id", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--notes", type=Path, required=True, help="Directory of .txt files to ingest.")
    parser.add_argument("--category", default="summary", help="Category label for stored notes.")
    args = parser.parse_args()

    manager = DoctorProfileManager()
    profile = manager.ensure(args.doctor_id, name=args.name)

    if not args.notes.exists():
        raise SystemExit(f"Notes directory not found: {args.notes}")

    files = sorted(args.notes.glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found in {args.notes}")

    for path in files:
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        manager.add_note(
            doctor_id=profile.doctor_id,
            content=content,
            title=path.stem.replace("_", " "),
            category=args.category,
        )
        print(f"Ingested {path}")

    print(f"Profile updated for {profile.doctor_id}. Notes stored: {len(manager.get_recent_notes(profile.doctor_id, 100))}")


if __name__ == "__main__":
    main()
