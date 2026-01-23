"""Configuration loading utilities for the GI Scribe app."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


CONFIG_FILE = Path("config.json")


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    input_device: Optional[int] = None
    silence_padding_ms: int = 300


@dataclass
class WhisperConfig:
    binary_path: str = "external/whisper.cpp/build/bin/Release/main.exe"
    model_path: str = "models/whisper/ggml-small.en.bin"
    faster_model: str = "medium.en"  # download name or local CTranslate2 dir
    engine: str = "auto"  # auto selects faster if available
    device: str = "auto"  # auto/cpu/cuda
    compute_type: str = "int8"
    beam_size: int = 5
    language: str = "en"
    threads: int = 8
    temperature: float = 0.0
    translate: bool = False
    extra_args: List[str] = field(default_factory=lambda: ["--print-progress"])


@dataclass
class SummarizerConfig:
    provider: str = "ollama"
    base_url: str = "http://127.0.0.1:11434"
    model: str = "medllama2"
    fallback_model: str = ""
    temperature: float = 0.15
    max_tokens: int = 500
    prompt_style: str = "Narrative"
    timeout_s: int = 600


@dataclass
class StorageConfig:
    root: str = "local_storage"
    retention_days: int = 90
    session_prefix: str = "session"
    auto_cleanup: bool = True


@dataclass
class UIConfig:
    theme: str = "DarkBlue3"
    default_summary_format: str = "Narrative"
    transcript_font: str = "Consolas"
    summary_font: str = "Segoe UI"
    max_visible_chars: int = 8000


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    summarizer: SummarizerConfig = field(default_factory=SummarizerConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "AppConfig":
        return AppConfig(
            audio=AudioConfig(**raw.get("audio", {})),
            whisper=WhisperConfig(**raw.get("whisper", {})),
            summarizer=SummarizerConfig(**raw.get("summarizer", {})),
            storage=StorageConfig(**raw.get("storage", {})),
            ui=UIConfig(**raw.get("ui", {})),
        )

    @classmethod
    def load(cls, path: Path = CONFIG_FILE) -> "AppConfig":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def save(self, path: Path = CONFIG_FILE) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def load_config() -> AppConfig:
    """Helper used across modules."""
    return AppConfig.load(CONFIG_FILE)
