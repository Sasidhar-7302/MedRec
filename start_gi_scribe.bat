@echo off
echo Starting GI Scribe with GPU acceleration...
set PATH=%~dp0cuda_libs;%PATH%
.venv\Scripts\python main.py
