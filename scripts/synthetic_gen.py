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

TRANSCRIPT_DIR = Path("data/synthetic/transcripts")
AUDIO_DIR = Path("data/synthetic/audio")
OLLAMA_URL = "http://localhost:11434/api/generate"

TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

def generate_transcript(scenario, doctor_role="Doctor", patient_role="Patient"):
    """Generate a realistic GI-specific transcript using the local LLM."""
    prompt = f"""[INST] <<SYS>>
You are a medical script writer for a high-fidelity clinical simulation used by medical students.
Your goal is to write a PURE DIALOGUE script between a GI Doctor and a Patient.

**CRITICAL RULES:**
1. **NO STAGE DIRECTIONS**: Do NOT include text in parentheses like (nods), (coughs), or (Doctor thinks). 
2. **NO META-TEXT**: Do NOT include introductions or conclusions about the script. The output must start with "Doctor: " or "Patient:".
3. **LENGTH**: The conversation must be very long and detailed (at least 80-100 turns). It should feel like a 10-minute visit.
4. **CLINICAL FLOW**:
    - **HPI**: Start with the reason for visit. Ask for pain location, quality (stabbing/burning/aching), duration, intensity (1-10), and triggers (food, lying down).
    - **ROS**: Detailed GI Review of Systems (nausea, vomiting, diarrhea, constipation, bloating, acid reflux, blood in stool, weight loss, appetite).
    - **History**: Ask about Past Medical/Surgical history (previous scopes, surgeries), Medications, and Family History (IBD, Celiac, Colorectal cancer).
    - **Social**: Diet (spicy/greasy/fiber), alcohol, stress levels, exercise.
    - **Assessment & Plan**: The doctor must summarize their thoughts to the patient and propose a clear plan (e.g., Blood tests, Endoscopy, Colonoscopy, dietary trials, or PPI/Biologic prescriptions).
5. **REALISM**: The doctor sounds professional and thorough. The patient sounds like a real person—sometimes rambling, using non-medical terms like "burning tummy" or "runs", and expressing concerns.

Scenario: {scenario}

Format:
{doctor_role}: [Text]
{patient_role}: [Text]
<</SYS>>
[/INST]"""

    payload = {
        "model": "medllama2",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.8,
            "num_predict": 4096,
            "num_ctx": 8192
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=400)
        response.raise_for_status()
        text = response.json().get("response", "").strip()
        # Post-clean: Remove any remaining lines with parentheses or meta-intro
        lines = text.split('\n')
        clean_lines = []
        started = False
        for line in lines:
            line = line.strip()
            if not started and not (line.startswith("Doctor:") or line.startswith("Patient:")):
                continue
            started = True
            # Strip anything in parentheses
            line = re.sub(r'\(.*?\)', '', line).strip()
            if line:
                clean_lines.append(line)
        return '\n'.join(clean_lines)
    except Exception as e:
        print(f"Error generating transcript: {e}")
        return None

async def synthesize_audio(transcript, doctor_voice, patient_voice, output_path):
    """Convert a multi-speaker transcript to audio using edge-tts."""
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
        
        if not text: continue
        
        voice = doctor_voice if 'doctor' in speaker else patient_voice
        
        # Process in chunks
        communicate = edge_tts.Communicate(text, voice)
        communicate_objects.append(communicate)

    with open(output_path, "wb") as f:
        for i, comm in enumerate(communicate_objects):
            if i > 0: await asyncio.sleep(0.05)
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])

async def process_batch(num_files=30):
    """Generate the full batch of synthetic data."""
    for i in range(num_files):
        case_id = f"SYNTH_{i+1:03d}"
        transcript_file = TRANSCRIPT_DIR / f"{case_id}.txt"
        audio_file = AUDIO_DIR / f"{case_id}.mp3"
        meta_file = TRANSCRIPT_DIR / f"{case_id}.json"
        
        if transcript_file.exists() and audio_file.exists():
            print(f"Skipping {case_id}, already exists.")
            continue

        print(f"Generating case {i+1}/{num_files}...")
        
        scenario = random.choice(GI_SCENARIOS)
        doc_key = random.choice(list(VOICES.keys()))
        pat_key = random.choice(list(VOICES.keys()))
        
        doctor_voice = VOICES[doc_key]
        patient_voice = VOICES[pat_key]
        
        transcript = generate_transcript(scenario)
        if not transcript or len(transcript) < 200:
            print(f"Failed to generate valid transcript for {case_id}")
            continue
            
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(transcript)
            
        metadata = {
            "case_id": case_id,
            "scenario": scenario,
            "doctor_voice": doctor_voice,
            "patient_voice": patient_voice,
            "voices": {"Doctor": doc_key, "Patient": pat_key}
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Synthesizing audio for {case_id} (Voices: {doc_key}, {pat_key})...")
        await synthesize_audio(transcript, doctor_voice, patient_voice, audio_file)
        print(f"Completed {case_id}.\n")

if __name__ == "__main__":
    import sys
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    asyncio.run(process_batch(count))
