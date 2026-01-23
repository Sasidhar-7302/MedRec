"""Audio recording helpers using sounddevice."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from .config import AudioConfig


class AudioRecorder:
    """Stream microphone input to a WAV file."""

    def __init__(self, config: AudioConfig):
        self.config = config
        self.logger = logging.getLogger("medrec.audio")
        self._stream: Optional[sd.InputStream] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._file: Optional[sf.SoundFile] = None
        self._stop_event = threading.Event()
        self._output_path: Optional[Path] = None
        self._last_file_size: Optional[int] = None
        self._active_sample_rate = self.config.sample_rate

    def start(self, output_path: Path) -> None:
        if self.is_recording:
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sample_rate = self._resolve_sample_rate()
        self._active_sample_rate = sample_rate
        self._file = sf.SoundFile(
            output_path,
            mode="w",
            samplerate=sample_rate,
            channels=self.config.channels,
            subtype="PCM_16",
        )
        # remember path so stop() can check size
        self._output_path = output_path
        self._stop_event.clear()
        self._queue = queue.Queue()
        device = self._normalize_device(self.config.input_device)
        self._stream = sd.InputStream(
            samplerate=sample_rate,
            channels=self.config.channels,
            dtype="int16",
            device=device,
            callback=self._on_audio_chunk,
        )
        self._stream.start()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

    def stop(self) -> None:
        if not self.is_recording:
            return
        if self._stream:
            self._stream.stop()
            self._stream.close()
        self._stop_event.set()
        self._queue.put(None)  # type: ignore[arg-type]
        if self._writer_thread:
            self._writer_thread.join(timeout=5)
        if self._file:
            self._file.close()
        # capture resulting file size for downstream checks
        try:
            if self._output_path and self._output_path.exists():
                self._last_file_size = self._output_path.stat().st_size
            else:
                self._last_file_size = 0
        except Exception:
            self._last_file_size = None
        self._reset()

    def _reset(self) -> None:
        self._stream = None
        self._writer_thread = None
        self._file = None
        self._stop_event.clear()
        self._output_path = None

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    def _on_audio_chunk(self, indata, frames, time, status) -> None:  # type: ignore[override]
        if status:
            print(f"[AudioRecorder] status: {status}")
        self._queue.put(indata.copy())

    def _writer_loop(self) -> None:
        assert self._file is not None
        while True:
            chunk = self._queue.get()
            if chunk is None:
                break
            self._file.write(chunk)
        self._file.flush()

    @property
    def last_file_size(self) -> Optional[int]:
        """Return size in bytes of the last recorded file (or None if unknown)."""
        return self._last_file_size

    def _resolve_sample_rate(self) -> int:
        desired = self.config.sample_rate
        device = self._normalize_device(self.config.input_device)
        if self._supports_rate(device, desired):
            return desired
        fallback = self._device_default_samplerate(device)
        if fallback and fallback != desired and self._supports_rate(device, fallback):
            self.logger.warning(
                "Input device rejected %s Hz. Falling back to %s Hz.",
                desired,
                fallback,
            )
            return int(fallback)
        raise RuntimeError(
            "Unable to open the selected microphone with the configured sample rate. "
            "Choose a different device or adjust audio.sample_rate."
        )

    def _supports_rate(self, device: Optional[int | str], rate: Optional[float]) -> bool:
        if not rate:
            return False
        try:
            sd.check_input_settings(device=device, samplerate=rate, channels=self.config.channels)
        except sd.PortAudioError:
            return False
        return True

    @staticmethod
    def _normalize_device(device: Optional[int | str]) -> Optional[int | str]:
        if device in (None, ""):
            return None
        if isinstance(device, str):
            try:
                return int(device)
            except ValueError:
                return device
        return device

    def _device_default_samplerate(self, device: Optional[int | str]) -> Optional[int]:
        try:
            if device is None:
                default_idx = sd.default.device[0]
                if default_idx is None or default_idx < 0:
                    return None
                info = sd.query_devices(default_idx)
            else:
                info = sd.query_devices(device)
            rate = info.get("default_samplerate")
            if rate:
                return int(rate)
        except Exception:
            return None
        return None


def list_input_devices() -> list[str]:
    """Return a list of available microphone device names."""
    devices = sd.query_devices()
    return [f"{idx}: {dev['name']}" for idx, dev in enumerate(devices) if dev["max_input_channels"] > 0]
