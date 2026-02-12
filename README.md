# GI Scribe

**Local-first AI Medical Scribe.**
GI Scribe listens to patient encounters, transcribes audio (Whisper), polishes the transcript (Llama 3), and generates structured clinical notes (Llama 3). All data remains 100% offline.

## 🚀 Key Features

*   **Three-Stage Pipeline:**
    1.  **Transcribe:** OpenAI Whisper (`large-v3`) for high-accuracy speech-to-text.
    2.  **Polish:** Llama 3 for verbatim correction (fixes "womiting" -> "vomiting" while preserving natural speech).
    3.  **Summarize:** Llama 3 for extraction of HPI, Assessment, and Plan.
*   **Zero-Hallucination:** Strict safeguards ensure "Not Documented" is returned if info is missing.
*   **Privacy First:** No cloud APIs. No data leaves your machine.

## 📂 Project Structure

*   `app/`: Core application logic (UI, Engines).
*   `data/`: Validation datasets (`GiAudiotest`, `GiTestValid`).
*   `models/`: Binary model files (GGML/GGUF).
*   `scripts/`: Utility scripts for benchmarking and maintenance.
*   `tests/`: Human-readable test scripts.
*   `docs/`: Architecture and User Guides.
*   `ARCHITECTURE.md`: Technical deep dive into the pipeline.

## 🛠️ Setup

### 1. Python Environment
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Whisper (C++ Backend)
```powershell
git clone https://github.com/ggerganov/whisper.cpp external\whisper.cpp
cd external\whisper.cpp
cmake -B build -DBUILD_SHARED_LIBS=OFF
cmake --build build --config Release
```
Download `ggml-large-v3.bin` into `models\whisper`.

### 3. Llama 3 (Ollama)
Install [Ollama](https://ollama.com/) and pull the model:
```powershell
ollama pull llama3
```
Ensure `ollama serve` is running.

### 4. Configuration
Copy `config.example.json` to `config.json`. Ensure `model` is set to `llama3`.

## 🧪 Verification

Run the full end-to-end pipeline test:
```powershell
python scripts/validate_full_pipeline.py
```
This will transcribe, polish, and summarize the sample files in `data/GiAudiotest`.

## ▶️ Running the App
```powershell
python main.py
```

## 📄 Documentation
See [ARCHITECTURE.md](ARCHITECTURE.md) for system design details.
