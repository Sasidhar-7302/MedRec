"""Local storage management."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .config import StorageConfig


@dataclass
class SessionArtifacts:
    session_dir: Path
    audio_path: Path
    transcript_path: Path
    summary_path: Path


class StorageManager:
    def __init__(self, config: StorageConfig):
        self.config = config
        self.root = Path(config.root)
        self.sessions_dir = self.root / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _session_directory(self) -> Path:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = f"{self.config.session_prefix}_{ts}"
        return self.sessions_dir / name

    def persist(
        self,
        audio_file: Path,
        transcript: str,
        summary: str,
        metadata: Optional[dict] = None,
    ) -> SessionArtifacts:
        session_dir = self._session_directory()
        session_dir.mkdir(parents=True, exist_ok=True)
        ts_name = datetime.now().strftime("recording_%Y%m%d_%H%M%S.wav")
        audio_dst = session_dir / ts_name
        shutil.copy2(audio_file, audio_dst)
        transcript_path = session_dir / "transcript.txt"
        transcript_path.write_text(transcript, encoding="utf-8")
        summary_path = session_dir / "summary.txt"
        summary_path.write_text(summary, encoding="utf-8")
        info = {
            "created_at": datetime.now().isoformat(),
            "audio_file": str(audio_dst),
            "transcriber_runtime_s": metadata.get("transcriber_runtime_s") if metadata else None,
            "summarizer_runtime_s": metadata.get("summarizer_runtime_s") if metadata else None,
            "whisper_command": metadata.get("whisper_command") if metadata else None,
        }
        (session_dir / "metadata.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        return SessionArtifacts(
            session_dir=session_dir,
            audio_path=audio_dst,
            transcript_path=transcript_path,
            summary_path=summary_path,
        )

    def purge_old_sessions(self) -> int:
        cutoff = datetime.now() - timedelta(days=self.config.retention_days)
        removed = 0
        for session_dir in self.sessions_dir.glob(f"{self.config.session_prefix}_*"):
            mtime = datetime.fromtimestamp(session_dir.stat().st_mtime)
            if mtime < cutoff:
                shutil.rmtree(session_dir, ignore_errors=True)
                removed += 1
        return removed
