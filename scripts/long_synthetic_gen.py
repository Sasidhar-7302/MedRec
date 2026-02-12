import asyncio
import json
import random
import os
import requests
import re
from pathlib import Path
import edge_tts

# Selected voices for diversity
VOICES = {
    "US_F": "en-US-AvaNeural",
    "US_M": "en-US-AndrewNeural",
    "UK_F": "en-GB-SoniaNeural",
    "UK_M": "en-GB-RyanNeural",
    "IN_F": "en-IN-NeerjaExpressiveNeural",
    "IN_M": "en-IN-PrabhatNeural",
    "AU_F": "en-AU-NatashaNeural",
    "AU_M": "en-AU-WilliamMultilingualNeural",
    "CA_M": "en-CA-LiamNeural",
    "ZA_F": "en-ZA-LeahNeural",
    "IE_M": "en-IE-ConnorNeural",
    "NG_F": "en-NG-EzinneNeural",
    "PH_M": "en-PH-JamesNeural"
}

GI_SCENARIOS = [
    "Chronic bloating and suspected Celiac disease follow-up. Discuss Gluten-free diet challenges.",
    "Acute Crohn's disease flare-up. Abdominal pain, 6-8 loose stools/day. Discuss biologics vs steroids.",
    "Persistent heartburn and acid reflux (GERD). Dysphagia concerns. Rule out Barrett's Esophagus.",
    "Ulcerative Colitis clinical remission check. Discussing tapering Prednisone.",
    "Biliary colic symptoms. Gallstones seen on US. Discussion of cholecystectomy vs observation.",
    "Chronic idiopathic constipation. Failing OTC laxatives. Discussing secretagogues like Linzess.",
    "Irritable bowel syndrome (IBS-D) triage. Discussion of Low FODMAP and stressors.",
    "Post-colonoscopy review: 3 small tubular adenomas found. Discussing 5-year surveillance.",
    "Jaundice and pruritus evaluation. Suspected PBC or PSC. Discussing MRCP.",
    "Eosinophilic Esophagitis focal dialogue. Impaction history. Reviewing biopsy results.",
    "Gastroparesis management. Bloating and fullness. Reglan vs small frequent meals.",
    "Nonalcoholic Fatty Liver Disease (NAFLD). Elevated ALT. Lifestyle and Fibroscan discussion.",
    "Small Intestinal Bacterial Overgrowth (SIBO). Positive breath test. Xifaxan course discussion.",
    "C. diff infection recurrence dialogue. Frequent diarrhea post-antibiotics. Dificid discussion.",
    "Microscopic colitis (collagenous/lymphocytic). Chronic watery diarrhea. Budesonide trial.",
    "Small bowel obstruction (partial). History of adhesions. Liquid diet and monitoring plan."
]

TRANSCRIPT_DIR = Path("data/synthetic_long/transcripts")
AUDIO_DIR = Path("data/synthetic_long/audio")
OLLAMA_URL = "http://localhost:11434/api/generate"

TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

def call_ollama(prompt):
    payload = {
        "model": "medllama2",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.8,
            "num_predict": 512, # Shorter for turns
            "num_ctx": 16384
        }
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"Ollama error: {e}")
        return ""

def generate_turn(history, role, scenario, stage_goal):
    prompt = f"""[INST] <<SYS>>
You are writing a clinical simulation dialogue between a Doctor and a Patient.
Scenario: {scenario}
Current Goal: {stage_goal}
Role to generate: {role}

**RULES:**
1. Generate ONE turn of dialogue for {role} ONLY.
2. Be VERY VERBOSE (3-5 sentences). 
3. NO STAGE DIRECTIONS (no text in parentheses).
4. Do NOT start with "{role}: ". Just provide the text.
5. Stay consistent with the conversation history.

HISTORY:
{chr(10).join(history[-10:])}
<</SYS>>
[/INST]"""
    
    response = call_ollama(prompt)
    # Clean any accidental labels or parentheses
    response = re.sub(r'\(.*?\)', '', response).strip()
    response = response.replace(f"{role}:", "").strip()
    return f"{role}: {response}"

def generate_long_transcript_iterative(scenario):
    stages = [
        ("Presentation & HPI", "The doctor asks about the primary complaint and explores the pain in detail.", 4),
        ("ROS & Previous History", "The doctor performs a Review of Systems and asks about medical history/meds.", 4),
        ("Social & Family History", "The doctor explores family history, diet, stress, and lifestyle factors.", 4),
        ("Assessment & Plan", "The doctor provides a detailed assessment and a multi-step plan.", 4)
    ]
    
    full_history = []
    
    for stage_name, stage_goal, turn_count in stages:
        print(f"  Stage: {stage_name}...")
        for _ in range(turn_count // 2):
            # Doctor's turn
            doc_turn = generate_turn(full_history, "Doctor", scenario, stage_goal)
            if doc_turn: full_history.append(doc_turn)
            
            # Patient's turn
            pat_turn = generate_turn(full_history, "Patient", scenario, stage_goal)
            if pat_turn: full_history.append(pat_turn)
            
    return '\n'.join(full_history)

async def synthesize_audio(transcript, doctor_voice, patient_voice, output_path):
    lines = transcript.split('\n')
    communicate_objects = []

    for line in lines:
        if not line or ':' not in line:
            continue
        
        parts = line.split(':', 1)
        if len(parts) < 2: continue
        speaker, text = parts
        speaker = speaker.strip().lower()
        text = text.strip()
        
        voice = doctor_voice if 'doctor' in speaker else patient_voice
        if not text: continue
        
        communicate = edge_tts.Communicate(text, voice)
        communicate_objects.append(communicate)

    with open(output_path, "wb") as f:
        for i, comm in enumerate(communicate_objects):
            if i > 0: await asyncio.sleep(0.05)
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])

async def process_batch(num_files=5):
    for i in range(num_files):
        case_id = f"LONG_SYNTH_{i+1:03d}"
        transcript_file = TRANSCRIPT_DIR / f"{case_id}.txt"
        audio_file = AUDIO_DIR / f"{case_id}.mp3"
        meta_file = TRANSCRIPT_DIR / f"{case_id}.json"
        
        if transcript_file.exists() and audio_file.exists():
            print(f"Skipping {case_id}, already exists.")
            continue

        print(f"\nProcessing {case_id}...")
        scenario = random.choice(GI_SCENARIOS)
        
        doc_key = random.choice(list(VOICES.keys()))
        pat_key = random.choice([k for k in VOICES.keys() if k != doc_key])
        
        doctor_voice = VOICES[doc_key]
        patient_voice = VOICES[pat_key]
        
        transcript = generate_long_transcript_iterative(scenario)
        if not transcript or len(transcript) < 1000:
            print(f"Failed to generate long transcript for {case_id}")
            continue
            
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(transcript)
            
        metadata = {
            "case_id": case_id,
            "scenario": scenario,
            "voices": {"Doctor": doc_key, "Patient": pat_key},
            "char_count": len(transcript)
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Synthesizing LONG audio for {case_id}...")
        await synthesize_audio(transcript, doctor_voice, patient_voice, audio_file)
        print(f"Completed {case_id}.")

if __name__ == "__main__":
    import sys
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    asyncio.run(process_batch(count))
