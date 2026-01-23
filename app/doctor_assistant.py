"""Doctor-facing coaching assistant built on the summarizer backend."""

from __future__ import annotations

from typing import List, Optional

from .config import SummarizerConfig
from .doctor_profiles import DoctorProfileManager
from .prompt_templates import build_doctor_chat_prompt
from .summarizer import OllamaSummarizer, SummaryResult


class DoctorAssistant:
    def __init__(self, config: SummarizerConfig, profile_manager: DoctorProfileManager):
        self.summarizer = OllamaSummarizer(config)
        self.profiles = profile_manager

    def respond(
        self,
        doctor_id: str,
        user_message: str,
        transcript: str = "",
        history: Optional[List[dict]] = None,
        max_context_notes: int = 3,
    ) -> SummaryResult:
        history = history or []
        profile_context = self.profiles.build_profile_context(doctor_id, max_notes=max_context_notes)
        history_text = format_history(history)
        prompt = build_doctor_chat_prompt(
            profile_context=profile_context,
            history=history_text,
            transcript=transcript,
            user_message=user_message,
        )
        result = self.summarizer.generate(prompt, temperature=0.2, max_tokens=400)
        self.profiles.record_interaction(doctor_id, user_message, result.summary)
        return result


def format_history(history: List[dict]) -> str:
    lines: List[str] = []
    for turn in history[-10:]:
        role = turn.get("role", "user").upper()
        content = turn.get("content", "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)
