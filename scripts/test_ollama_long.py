import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"

prompt = """[INST] <<SYS>>
You are a medical script writer. Write a **VERY LONG** PURE DIALOGUE script for a medical simulation.
STAGE: Presentation & HPI
INSTRUCTION: Start the visit. Ask for the reason for coming in, pain details (location, quality, intensity), and recent triggers. Be extremely wordy and detailed. Write at least 25 back-and-forth exchanges.

**CRITICAL RULES:**
1. **NO STAGE DIRECTIONS** (no text in parentheses).
2. **PURE DIALOGUE ONLY**.
3. **EXTREMELY VERBOSE**: Each response should be 3-4 sentences.
4. **LENGTH**: Write at least 20 back-and-forth exchanges for THIS STAGE ALONE.

Output format:
Doctor: [Text]
Patient: [Text]
<</SYS>>
[/INST]"""

payload = {
    "model": "medllama2",
    "prompt": prompt,
    "stream": False,
    "options": {
        "temperature": 0.8,
        "num_predict": 4096,
        "num_ctx": 16384
    }
}

try:
    response = requests.post(OLLAMA_URL, json=payload, timeout=600)
    response.raise_for_status()
    print(response.json().get("response", "").strip())
except Exception as e:
    print(f"Error: {e}")
