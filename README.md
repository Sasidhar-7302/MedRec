# GI Scribe (Windows Prototype)

Local-first desktop application for gastroenterology dictations. Records audio, transcribes it with `whisper.cpp`, summarizes the transcript with a local Med-LLaMA 2 model (via Ollama), and stores session artifacts on disk for up to 90 days.

## Requirements

- Windows 10/11, 24 GB RAM recommended (8 GB minimum for transcription-only).
- Python 3.11 (64-bit) with `pip` and `venv`.
- `git`, `cmake`, and a C++ build toolchain (e.g., Visual Studio Build Tools) to compile `whisper.cpp` and `llama.cpp`.
- USB microphone with reasonable noise filtering.

## Setup

1. **Python environment**
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **whisper.cpp**
   ```powershell
   git clone https://github.com/ggerganov/whisper.cpp external\\whisper.cpp
   cd external\\whisper.cpp
   cmake -B build -DBUILD_SHARED_LIBS=OFF
   cmake --build build --config Release
   ```
   Download a GGML model (e.g., `ggml-small.en.bin`) into `models\\whisper`. Run a sanity test:
   ```powershell
   .\\build\\bin\\Release\\whisper-cli.exe -m ..\\..\\models\\whisper\\ggml-small.en.bin -f samples\\jfk.wav -otxt
   ```

3. **MedLlama2 via Ollama**
   ```powershell
   winget install ollama.ollama
   ollama pull medllama2:latest
   ```
   Ensure the daemon is running (`ollama serve`). The app talks to `http://localhost:11434`.  
   If you see `Only one usage of each socket...` it means Ollama is already running, so you can ignore the warning.  
   The desktop app will try to auto-start the service when it's missing, and a “Launch Summarizer Service” button is available under the status cards for manual control.

4. **Configuration**
   Copy `config.example.json` → `config.json` and update paths to the compiled `whisper.cpp` binary, GGML model, and retention policy.

5. **Doctor profile asset (optional but recommended)**
   ```powershell
   New-Item -ItemType Directory -Path assets -Force
   curl https://www.gigeorgia.com/app/uploads/2024/11/GISpecialists_Swaroop-Pendyala.jpg -o assets\dr_pendyala.jpg
   ```

## Improving Accuracy

For maximum accuracy, especially for speaker differentiation and HPI/Assessment extraction, see:
- **[Accuracy Improvements Guide](docs/README_ACCURACY_IMPROVEMENTS.md)** - Quick reference
- **[Google Colab Fine-Tuning Guide](docs/GOOGLE_COLAB_FINETUNING_GUIDE.md)** - Step-by-step fine-tuning
- **[Accuracy Improvement Guide](docs/ACCURACY_IMPROVEMENT_GUIDE.md)** - Comprehensive strategies

**Quick Start:** Fine-tune models in Google Colab to achieve 95%+ accuracy for HPI extraction and Assessment.
   The UI automatically picks up this image; if it is missing, a placeholder avatar is shown.

## 📁 Project Structure

The project is organized as follows:

- `app/`: Main application source code
  - `ui_redesigned.py`: Main GUI implementation
  - `transcriber.py`: Whisper integration
  - `summarizer.py`: Ollama integration
- `docs/`: Documentation and guides (`VERIFICATION_GUIDE.md`, etc.)
- `reports/`: Test reports and performance analysis
- `tests/`: Automated test suite
- `data/`: Training and validation datasets
- `models/`: Local model storage
- `external/`: External binaries and tools

## 🧪 Testing

### Quick Test
Run the quick verification suite:
```bash
python -m tests.test_quick
```

### Full Test Suite
Run the comprehensive tests:
```bash
python -m tests.test_comprehensive
```

### Accuracy Check
Verify model accuracy:
```bash
python -m scripts.verify_accuracy
```

See [docs/VERIFICATION_GUIDE.md](docs/VERIFICATION_GUIDE.md) for more details.

## 6. Run the prototype
   ```powershell
   python main.py
   ```

## Packaging

PyInstaller spec file (`gi_scribe.spec`) is provided. Build with:
```powershell
pyinstaller gi_scribe.spec
```
The resulting executable bundles the Python runtime; the `models` and `external` binaries remain alongside the EXE for now.

## Workflow Highlights

- **Record → Transcribe → Summarize** pipeline with offline components only.
- **Local storage** keeps audio/transcript/summary bundles under `local_storage\\sessions`.
- Automatic purge runs at startup; schedule `python -m app.cleanup` weekly via Windows Task Scheduler for unattended cleanup.
- **UI** built with PySide6 to mirror the mobile-first cards shown in the reference mocks, including Dr. Pendyala’s profile photo, live recording timer, and recent-session history.

See `docs/setup.md` and `docs/workflow.md` for deeper instructions and rollout plan.
