# MedRec Architecture & Model Overview

Last updated: 2025-11-17  
Author: Codex assistant

## 1. Product Summary
MedRec is a HIPAA-friendly, offline-first desktop assistant that records GI specialist dictations, produces a transcript, and generates structured EHR-ready summaries (Findings / Assessment / Plan / Medications / Follow-up). The stack combines local audio capture, Whisper-based ASR, and an Ollama-hosted MedLlama summarizer. All artifacts are stored in `local_storage/` with optional cleanup after 90 days.

## 2. Model Stack

| Capability | Default Model | Notes |
|------------|---------------|-------|
| Automatic Speech Recognition | **Whisper**<br>- CLI: `ggml-small.en.bin` via whisper.cpp<br>- Low-latency: Faster-Whisper `medium.en` (default) | `config.whisper.engine` defaults to `auto`, which prefers Faster-Whisper if installed and falls back to CLI otherwise. Medium.en int8 fits in ~3 GB RAM and streams partial text to the UI in near real time. |
| Summarization | **MedLlama2** via Ollama | Primary: `medllama2:7b-instruct-q4_K_M` (fast on CPU)<br>Fallback: `medllama2:13b-instruct-q4` (auto retry if 7B fails). Temperature 0.15, max 500 tokens. |
| Prompt Template | Narrative / SOAP in `app/prompt_templates.py` | Both prompts seed GI-specific terminology (Crohn's, Barrett's, vedolizumab, etc.) and enforce structured headings. |
| Terminology Corrections | `app/terminology.apply_corrections` | Deterministic replacements post-transcription (add new GI terms here for accuracy). |
| Vocabulary context | `data/gi_terms.txt` | Central glossary used by Whisper (initial prompt), the summarizer prompts, and correction hints. Edit this file to inject additional terms without touching code. |

### Model Locations
- Whisper CLI weights live under `models/whisper/`.
- Faster-Whisper downloads to `models/faster-whisper/`.
- Ollama pulls models into `%LOCALAPPDATA%\Ollama\models`.

## 3. Application Architecture

```
┌──────────────┐
│ PySide6 UI   │
│ (app/ui_*)   │
└─────┬────────┘
      │
      │ 1. Start Recording / Load Audio
      ▼
┌──────────────┐
│ AudioRecorder│ (app/audio.py)
│ - sounddevice│
└─────┬────────┘
      │ WAV path
      ▼
┌──────────────────┐
│ WhisperTranscriber│ (app/transcriber.py)
│ - CLI or faster   │
└─────┬────────────┘
      │ transcript text + runtime
      ▼
┌────────────────┐
│ OllamaSummarizer│ (app/summarizer.py)
│ - medllama2     │
└─────┬──────────┘
      │ summary text + runtime + model used
      ▼
┌───────────────────┐
│ StorageManager     │ (app/storage.py)
│ - persists audio   │
│ - transcript       │
│ - summary + metadata (timings, whisper command) │
└───────────────────┘
```

The UI keeps a thread pool (`ThreadPoolExecutor`) to run ASR and summarization off the GUI thread. Results return through a queue, enabling live partial transcripts when Faster-Whisper is active. Service-status cards periodically call health checks (`WhisperTranscriber` binary existence + `OllamaSummarizer.health_check`).

## 4. Current Stage (2025-11-17)
| Area | Status | Notes |
|------|--------|-------|
| UI redesign | ✅ Complete | Modern split layout with hero, folders, live transcript + summary panels. |
| Audio capture | ✅ Complete | Supports selectable devices, sample-rate auto fallback, streaming writer thread. |
| Transcription | ✅ Dual-mode | CLI works; Faster-Whisper auto-download + streaming enabled. Needs CUDA for best speed but runs on CPU. |
| Summarization | ✅ Primary & fallback | 13B default, 7B fallback. Logs record model used + runtime. |
| Storage / Recents | ✅ | Sessions persisted per dictation with metadata and auto-cleanup option. |
| Packaging | ⚠ Pending | PyInstaller spec exists but final EXE not built yet. |
| Personalization | ⚠ Planned | No adaptive fine-tuning yet beyond terminology corrections. |

## 5. Performance Snapshot
Recent log excerpts (`python main.py` on Windows laptop, CPU-only):

| Stage | Model Config | Duration | Context |
|-------|--------------|----------|---------|
| Transcription | Faster-Whisper `medium.en` (int8) | **~18 s** | CPU-only Intel i5 desktop with 24 GB RAM. CLI fallback remains ~33 s. CUDA would cut this to <5 s. |
| Summarization | MedLlama2 7B q4 | **40–55 s** | Same dictation, 1187-character output. 13B fallback still available (~150 s). |

### Bottlenecks & Next Steps
1. **Stay on Faster-Whisper** (`pip install -r requirements.txt`). Override `device: "cuda"` + `compute_type: "int8_float16"` when you have a GPU.
2. **Use the 7B summarizer for daily work** (default). Switch to 13B only when you need extra detail.
3. **Generate synthetic GI data** (`scripts/generate_synthetic_gi_data.py`) and follow the `docs/FINETUNING_PLAYBOOK.md` to fine-tune both models.
4. **Optional**: implement chunk-level streaming if you need <5 s live captions; this requires recorder refactor + incremental prompt updates.

## 6. Present Performance Checklist
- [x] Audio capture stable (no PaError -9997, device filtering in UI).
- [x] Logs include `transcription_start/complete` and `summary_start/complete` timing lines for every session.
- [ ] Need GPU-backed benchmark to reach <10 s summarizer latency.
- [ ] Need automated accuracy evaluation (WER/ROUGE) with real GI corpora.

Use this document to brief stakeholders on the technical foundation, the models in play, and the optimization roadmap. Update the performance table whenever you run new benchmarks or swap models.

For detailed fine-tuning steps, synthetic data instructions, and training commands, see `docs/FINETUNING_PLAYBOOK.md`.
