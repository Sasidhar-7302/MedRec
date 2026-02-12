"""Transcription engines: whisper.cpp CLI plus faster-whisper for low-latency streaming."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

# Add CUDA libraries to path for faster-whisper on Windows
CUDA_LIBS_DIR = Path(__file__).parent.parent / "cuda_libs"
if CUDA_LIBS_DIR.exists():
    os.environ["PATH"] = str(CUDA_LIBS_DIR) + os.pathsep + os.environ.get("PATH", "")

from .config import WhisperConfig
from .gi_terms import build_gi_hint
from .gi_post_processor import process_transcription


ProgressCallback = Optional[Callable[[str], None]]


@dataclass
class TranscriptionResult:
    text: str
    runtime_s: float
    command: List[str] = field(default_factory=list)
    output_path: Optional[Path] = None
    segments: List[str] = field(default_factory=list)


class WhisperTranscriber:
    """Wrapper that can call whisper.cpp CLI or faster-whisper based on config."""

    def __init__(self, config: WhisperConfig):
        self.config = config
        self.logger = logging.getLogger("medrec.transcriber")
        self._faster_model = None
        self._engine = self._resolve_engine()

    def transcribe(self, audio_path: Path, progress_cb: ProgressCallback = None) -> TranscriptionResult:
        # Check for diarization request
        if getattr(self.config, "diarization", None) and self.config.diarization.enabled:
             if self.config.diarization.provider == "whisperx":
                 return self._transcribe_diarized(audio_path, progress_cb)
        
        if self._engine == "faster":
            return self._transcribe_faster(audio_path, progress_cb)
        return self._transcribe_cli(audio_path)

    def _transcribe_diarized(self, audio_path: Path, progress_cb: ProgressCallback) -> TranscriptionResult:
        """Transcribe using GIEar (WhisperX) with speaker diarization."""
        try:
            from .diarizer import Diarizer
        except ImportError as e:
            raise RuntimeError(f"GIEar (Diarizer) import failed: {e}")

        self.logger.info("Starting GIEar Diarization on %s", audio_path)
        start = time.perf_counter()
        
        # Initialize Diarizer (it loads models on demand)
        # Note: We might want to cache this instance if possible, but for now allow re-init per call or trust Diarizer to handle caching
        # Actually Diarizer class in app/diarizer.py re-loads models. 
        # Refinement: We should probably instantiate Diarizer once if possible, or just use it as is.
        # For this instruction, we'll instantiate it here.
        model_name = self.config.faster_model or "large-v3"
        diarizer = Diarizer(device=self.config.device if self.config.device != "auto" else "cuda", 
                           compute_type=self.config.compute_type,
                           model_name=model_name)
        
        # Run pipeline with medical biasing
        prompt_hint = build_gi_hint(max_terms=60)
        segments = diarizer.process_audio(
            str(audio_path),
            min_speakers=self.config.diarization.min_speakers,
            max_speakers=self.config.diarization.max_speakers,
            initial_prompt=prompt_hint
        )
        
        # Format output
        formatted_lines = []
        for seg in segments:
            start_fmt = time.strftime('%H:%M:%S', time.gmtime(seg['start']))
            end_fmt = time.strftime('%H:%M:%S', time.gmtime(seg['end']))
            line = f"[{start_fmt} - {end_fmt}] {seg['speaker']}: {seg['text']}"
            formatted_lines.append(line)
            if progress_cb:
                progress_cb(line)
                
        full_text = "\n".join(formatted_lines)
        runtime = time.perf_counter() - start
        
        # Cleanup
        diarizer.cleanup()
        
        return TranscriptionResult(
            text=full_text,
            runtime_s=runtime,
            command=["diarizer", "whisperx"],
            output_path=audio_path,
            segments=[s['text'] for s in segments] # Store raw text segments
        )


    def _resolve_engine(self) -> str:
        requested = (self.config.engine or "auto").lower()
        if requested == "cli":
            return "cli"
        faster_ready = self._faster_runtime_available()
        if requested == "faster":
            if faster_ready:
                return "faster"
            self.logger.warning("faster-whisper requested but package missing. Falling back to CLI.")
            return "cli"
        # auto mode prefers faster if possible
        return "faster" if faster_ready else "cli"

    @staticmethod
    def _faster_runtime_available() -> bool:
        try:
            import faster_whisper  # noqa: F401

            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------ whisper.cpp CLI
    def _transcribe_cli(self, audio_path: Path) -> TranscriptionResult:
        binary = Path(self.config.binary_path)
        model = Path(self.config.model_path)
        if not binary.exists():
            raise FileNotFoundError(f"whisper.cpp binary not found: {binary}")
        if not model.exists():
            raise FileNotFoundError(f"Whisper model not found: {model}")
        normalized_args = self._normalize_extra_args(self.config.extra_args)
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "whisper_out"
            cmd = [
                str(binary),
                "-m",
                str(model),
                "-f",
                str(audio_path),
                "-otxt",
                "-of",
                str(base),
                "-l",
                self.config.language,
                "-t",
                str(self.config.threads),
            ]
            if self.config.translate:
                cmd.append("--translate")
            if self.config.temperature is not None:
                cmd.extend(["--temperature", str(self.config.temperature)])
            if normalized_args:
                cmd.extend(normalized_args)
            prompt_hint = build_gi_hint(max_terms=60)
            if prompt_hint:
                cmd.extend(["--prompt", prompt_hint])
            start = time.perf_counter()
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            runtime = time.perf_counter() - start
            if proc.returncode != 0:
                combined = (proc.stderr or "") + "\n" + (proc.stdout or "")
                raise RuntimeError(f"whisper.cpp failed (returncode={proc.returncode}): {combined}")
            txt_path = Path(f"{base}.txt")
            if txt_path.exists():
                text = txt_path.read_text(encoding="utf-8")
            else:
                stdout = (proc.stdout or "").strip()
                stderr = (proc.stderr or "").strip()
                if stderr:
                    self.logger.warning(
                        "whisper.cpp did not emit a .txt file; stderr=%s cmd=%s", stderr, cmd
                    )
                if stdout:
                    txt_path.write_text(stdout, encoding="utf-8")
                text = stdout
            clean = text.strip()
            return TranscriptionResult(
                text=clean,
                runtime_s=runtime,
                command=cmd,
                output_path=txt_path,
                segments=[clean] if clean else [],
            )

    # ------------------------------------------------------------------ faster-whisper engine
    def _ensure_faster_model(self):
        if self._faster_model is not None:
            return self._faster_model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "faster-whisper is not installed. Run `pip install faster-whisper` in your environment."
            ) from exc

        device = self._select_device(self.config.device)
        compute = self.config.compute_type or "int8"
        model_source = self.config.faster_model or self.config.model_path
        download_root = Path("models") / "faster-whisper"
        download_root.mkdir(parents=True, exist_ok=True)
        self.logger.info(
            "Loading faster-whisper model %s on %s (%s)", model_source, device, compute
        )
        self._faster_model = WhisperModel(
            model_source,
            device=device,
            compute_type=compute,
            download_root=str(download_root),
        )
        return self._faster_model

    def _transcribe_faster(self, audio_path: Path, progress_cb: ProgressCallback) -> TranscriptionResult:
        model = self._ensure_faster_model()
        start = time.perf_counter()
        prompt_hint = build_gi_hint(max_terms=60)
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=self.config.language,
            temperature=self.config.temperature or 0.0,
            beam_size=self.config.beam_size,
            vad_filter=False,  # Disable VAD to avoid removing all audio
            initial_prompt=prompt_hint or None,
        )
        collected: List[str] = []
        for segment in segments_iter:
            text = segment.text.strip()
            if not text:
                continue
            collected.append(text)
            if progress_cb:
                progress_cb(" ".join(collected).strip())
        runtime = time.perf_counter() - start
        full_text = " ".join(collected).strip()
        # Apply GI-specific post-processing for accuracy
        processed_text = process_transcription(full_text)
        return TranscriptionResult(
            text=processed_text,
            runtime_s=runtime,
            command=["faster-whisper", self.config.faster_model or self.config.model_path],
            output_path=audio_path,
            segments=collected,
        )

    @staticmethod
    def _select_device(preferred: Optional[str]) -> str:
        preference = (preferred or "auto").lower()
        if preference in {"cpu", "cuda"}:
            return preference
        try:
            import torch

            if torch.cuda.is_available():  # pragma: no cover
                return "cuda"
        except Exception:
            pass
        return "cpu"

    @staticmethod
    def _normalize_extra_args(args: Optional[List[str]]) -> List[str]:
        normalized: List[str] = []
        if not args:
            return normalized
        for arg in args:
            if arg.startswith("--"):
                normalized.append(arg.replace("_", "-"))
            else:
                normalized.append(arg)
        return normalized
