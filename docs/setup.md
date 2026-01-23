# Environment Setup (Windows 10/11)

## 1. System Prerequisites

- Windows 10/11 Pro
- 24 GB RAM (minimum 16 GB for full workflow; 8 GB for transcription-only)
- 50 GB free SSD space
- USB microphone (cardioid, built-in noise gate preferred)
- Administrative rights to install Python/VS Build Tools/Ollama

## 2. Python 3.11

1. Download the 64-bit installer: <https://www.python.org/downloads/release/python-3110/>
2. During installation:
   - ✔ Add Python to PATH
   - ✔ Install for all users

Verify:
```powershell
py --list
py -3.11 --version
```

## 3. Virtual Environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Audio Stack

- `sounddevice` uses WASAPI by default. Run `python -m sounddevice` to list input devices and update `config.json` if you want to pin to a specific microphone.
- Optional: Install Voicemeeter or RTX Voice for noise reduction (not required).

## 5. whisper.cpp

```powershell
git clone https://github.com/ggerganov/whisper.cpp external\whisper.cpp
cmake -S external\whisper.cpp -B external\whisper.cpp\build -DCMAKE_BUILD_TYPE=Release
cmake --build external\whisper.cpp\build --config Release
```

Download a model:
```powershell
Invoke-WebRequest -Uri https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin -OutFile models\whisper\ggml-small.en.bin
```

Record a short sanity clip:
```powershell
python scripts\record_5s.py sanity.wav
external\whisper.cpp\build\bin\Release\whisper-cli.exe -m models\whisper\ggml-small.en.bin -f sanity.wav -otxt
```

## 6. llama.cpp / Ollama

Simplest path is Ollama:
```powershell
winget install ollama.ollama
ollama serve   # runs in a background terminal
ollama pull medllama2:latest
```
The `app.summarizer.OllamaSummarizer` talks to `http://localhost:11434/api/generate`.  
If `ollama serve` reports `Only one usage of each socket...`, the daemon is already running—no further action is required. The desktop app also exposes a “Launch Summarizer Service” button near the status cards and will auto-launch the daemon if it detects it isn't running.

For direct llama.cpp usage (optional), build `llama.cpp` similarly and point the config to your runner script.

## 7. Configuration & Assets

Copy template:
```powershell
Copy-Item config.example.json config.json
```

Fields to edit:
- `whisper.binary_path`
- `whisper.model_path`
- `summarizer.model`
- `storage.retention_days`

Optional (for branded UI):
```powershell
New-Item -ItemType Directory -Path assets -Force
curl https://www.gigeorgia.com/app/uploads/2024/11/GISpecialists_Swaroop-Pendyala.jpg -o assets\dr_pendyala.jpg
```

## 8. Running

```powershell
python main.py
```

## 9. Packaging

```powershell
pyinstaller gi_scribe.spec --noconfirm
```

Deliverable: `dist\GI_Scribe\GI_Scribe.exe`. Bundle `external\` binaries + `models\` folder and provide script to place them next to the EXE.

## 10. Scheduled Cleanup

Create a weekly task:

```powershell
SchTasks /Create /SC WEEKLY /D SUN /TN "GI Scribe Cleanup" /TR "\"%USERPROFILE%\\AppData\\Local\\Programs\\Python\\Python311\\python.exe\" \"C:\\Path\\To\\MedRec\\app\\cleanup.py\"" /ST 02:00
```

The script purges session folders older than the configured retention window.
