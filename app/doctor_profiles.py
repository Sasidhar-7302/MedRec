"""Per-doctor profile and preference management."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


@dataclass
class DoctorNote:
    title: str
    content: str
    category: str = "summary"
    created_at: str = field(default_factory=_now_iso)


@dataclass
class DoctorProfile:
    doctor_id: str
    name: str
    preferences: Dict[str, str] = field(default_factory=dict)
    vocabulary: List[str] = field(default_factory=list)
    notes: List[DoctorNote] = field(default_factory=list)
    last_updated: str = field(default_factory=_now_iso)

    def to_json(self) -> Dict:
        data = asdict(self)
        data["notes"] = [asdict(note) for note in self.notes]
        return data

    @classmethod
    def from_json(cls, payload: Dict) -> "DoctorProfile":
        notes = [DoctorNote(**note) for note in payload.get("notes", [])]
        profile = cls(
            doctor_id=payload["doctor_id"],
            name=payload.get("name", payload["doctor_id"]),
            preferences=payload.get("preferences", {}),
            vocabulary=payload.get("vocabulary", []),
            notes=notes,
            last_updated=payload.get("last_updated", _now_iso()),
        )
        return profile


class DoctorProfileManager:
    def __init__(self, root: Path = Path("data/doctor_profiles")) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, doctor_id: str) -> Path:
        return self.root / f"{doctor_id}.json"

    def load(self, doctor_id: str) -> Optional[DoctorProfile]:
        path = self._path(doctor_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return DoctorProfile.from_json(data)

    def ensure(self, doctor_id: str, name: Optional[str] = None) -> DoctorProfile:
        profile = self.load(doctor_id)
        if profile is None:
            profile = DoctorProfile(doctor_id=doctor_id, name=name or doctor_id)
            self.save(profile)
        elif name and profile.name != name:
            profile.name = name
            profile.last_updated = _now_iso()
            self.save(profile)
        return profile

    def save(self, profile: DoctorProfile) -> None:
        profile.last_updated = _now_iso()
        self._path(profile.doctor_id).write_text(
            json.dumps(profile.to_json(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def add_vocabulary(self, doctor_id: str, terms: List[str]) -> DoctorProfile:
        profile = self.ensure(doctor_id)
        for term in terms:
            clean = term.strip()
            if clean and clean not in profile.vocabulary:
                profile.vocabulary.append(clean)
        self.save(profile)
        return profile

    def add_note(self, doctor_id: str, content: str, title: Optional[str] = None, category: str = "summary") -> None:
        if not content.strip():
            return
        profile = self.ensure(doctor_id)
        note = DoctorNote(title=title or f"{category.title()} note", content=content.strip(), category=category)
        profile.notes.append(note)
        # Keep the most recent 50 notes to limit file size
        profile.notes = profile.notes[-50:]
        self.save(profile)

    def get_recent_notes(self, doctor_id: str, limit: int = 3) -> List[DoctorNote]:
        profile = self.load(doctor_id)
        if not profile:
            return []
        return profile.notes[-limit:]

    def build_profile_context(self, doctor_id: str, max_notes: int = 3) -> str:
        profile = self.ensure(doctor_id)
        lines = [
            f"Doctor: {profile.name} ({profile.doctor_id})",
        ]
        if profile.preferences:
            pref_lines = "; ".join(f"{k}: {v}" for k, v in profile.preferences.items())
            lines.append(f"Preferences: {pref_lines}")
        if profile.vocabulary:
            vocab = ", ".join(profile.vocabulary[:30])
            lines.append(f"Preferred terminology: {vocab}")
        notes = profile.notes[-max_notes:]
        if notes:
            formatted = []
            for idx, note in enumerate(notes, 1):
                formatted.append(f"{idx}. [{note.category}] {note.title}: {note.content[:400]}")
            lines.append("Recent approved summaries:\n" + "\n".join(formatted))
        return "\n".join(lines)

    def record_interaction(self, doctor_id: str, prompt: str, response: str) -> None:
        profile = self.ensure(doctor_id)
        profile.preferences.setdefault("last_request", prompt[:200])
        profile.preferences["last_response_chars"] = str(len(response))
        self.save(profile)
