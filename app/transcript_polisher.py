"""
Transcript Polisher
===================
Uses a Language Model (Llama 3) to "proofread" raw ASR transcripts.
Corrects phonetic errors, stutters, and grammar while maintaining verbatim fidelity.
"""

import logging
import time
import requests
import re
from typing import Optional
from dataclasses import dataclass
from .config import SummarizerConfig

@dataclass
class PolishingResult:
    original_text: str
    polished_text: str
    runtime_s: float
    model_used: str

POLISH_PROMPT = """[INST] <<SYS>>
You are a medical transcription editor. Your job is to fix phonetic errors and remove stutters.
**RULES:**
1. Output the EXACT same number of lines as the input.
2. Maintain all [timestamps] and Speaker: labels.
3. Fix medical typos (e.g. "hematezia" -> "hematochezia").
4. Remove stutters (e.g. "I-I-I think" -> "I think").
5. DO NOT SUMMARIZE. Output line-by-line.
<</SYS>>

Example Input:
[00:01] SPEAKER_00: So, um, tell me about the pain.
[00:04] SPEAKER_01: It horts right here in my, uh, bellie button area.

Example Output:
[00:01] SPEAKER_00: So, tell me about the pain.
[00:04] SPEAKER_01: It hurts right here in my belly button area.

Input Transcript:
{transcript}

Polished Output:
[/INST]"""

class TranscriptPolisher:
    def __init__(self, config: SummarizerConfig):
        self.config = config
        self.logger = logging.getLogger("medrec.transcript_polisher")
        self.max_retries = 2

    @property
    def _endpoint(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/api/generate"

    def polish(self, transcript: str) -> PolishingResult:
        """
        Polishes transcript by isolating the text of each turn,
        sending it to LLM for cleaning, and re-assembling.
        Passes multiple turns in a structured list to the LLM for context.
        """
        start_time = time.perf_counter()
        
        # 1. Parse Transcript into (Header, Text) tuples
        # Regex to find "[00:00] SPEAKER_XX:" patterns
        pattern = re.compile(r"^(\[\d{2}:\d{2}.*?\]\s*SPEAKER_\d+:)(.*)$", re.MULTILINE)
        
        lines = transcript.strip().split('\n')
        parsed_lines = []
        for line in lines:
            match = pattern.match(line)
            if match:
                parsed_lines.append({"header": match.group(1), "text": match.group(2).strip(), "is_speech": True})
            else:
                # Keep non-speech lines (like empty lines or noise) as is
                parsed_lines.append({"header": "", "text": line, "is_speech": False})
        
        # 2. Batch Polish Speech Lines
        # We process in batches of 10 to maintain context but enforce separation
        batch_size = 10
        speech_indices = [i for i, x in enumerate(parsed_lines) if x["is_speech"]]
        
        self.logger.info(f"Polishing {len(speech_indices)} speech turns...")
        
        for i in range(0, len(speech_indices), batch_size):
            batch_idxs = speech_indices[i:i+batch_size]
            
            # Create a map of PromptID -> RealID
            # PromptIDs will be 1, 2, 3... 10
            id_map = {k+1: v for k, v in enumerate(batch_idxs)}
            
            # Construct Prompt Input
            input_text = ""
            for prompt_id, real_idx in id_map.items():
                input_text += f"Line {prompt_id}: {parsed_lines[real_idx]['text']}\n"
            
            prompt = f"""[INST] <<SYS>>
You are a Medical Transcription Editor.
Your Goal: Fix ONLY phonetic errors and misspelled medical terms.
Rules:
1. Do NOT remove stutters (um, ah, like). Keep them exactly as is.
2. Do NOT fix grammar. Keep original phrasing.
3. Do NOT summarize.
4. Output Format: "LineNumber: Text"
<</SYS>>

Input:
{input_text}

Output:
[/INST]"""
            
            try:
                response = self._invoke_model(prompt)
                
                output_lines = response.strip().split('\n')
                updates_count = 0
                
                for out_line in output_lines:
                    # Robust Regex
                    m = re.match(r"^\D*(\d+)[:.]\s*(.*)", out_line)
                    if m:
                        prompt_id = int(m.group(1))
                        cleaned_text = m.group(2).strip()
                        
                        # Remove **bold** markers
                        cleaned_text = cleaned_text.replace("**", "")
                        
                        # Check if prompt_id exists in our map
                        if prompt_id in id_map:
                             real_idx = id_map[prompt_id]
                             
                             # Extra safety: Ensure we don't accidentally preserve the "Line X:" prefix inside the text
                             # If the model output "Line 1: Line 1: Text", regex captured "Line 1: Text"
                             # Recursive strip
                             sub_m = re.match(r"^\D*(\d+)[:.]\s*(.*)", cleaned_text)
                             if sub_m:
                                 cleaned_text = sub_m.group(2).strip()

                             parsed_lines[real_idx]["text"] = cleaned_text
                             updates_count += 1
                
                if updates_count == 0:
                     self.logger.warning(f"Batch {i}: No lines parsed from response. Response snippet: {response[:100]}")
                else:
                     self.logger.info(f"Batch {i}: Updated {updates_count}/{len(batch_idxs)} lines.")

            except Exception as e:
                self.logger.error(f"Batch polish failed: {e}")
        
        # 3. Reassemble
        final_lines = []
        for item in parsed_lines:
            if item["is_speech"]:
                final_lines.append(f"{item['header']} {item['text']}")
            else:
                final_lines.append(item['text'])
                
        final_text = "\n".join(final_lines)
        runtime = time.perf_counter() - start_time
        
        return PolishingResult(
            original_text=transcript,
            polished_text=final_text,
            runtime_s=runtime,
            model_used=self.config.model
        )

    def _invoke_model(self, prompt: str) -> str:
        """Invoke the model (Reused logic from TwoPassSummarizer)."""
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1, # Low temp for fidelity
                "num_ctx": getattr(self.config, "context_window", 8192), # Ensure large context
                "num_predict": -1, # Unlimited generation
            },
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self._endpoint,
                    json=payload,
                    timeout=600, # Long timeout for full polish
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                data = response.json()
                text = data.get("response", "").strip()
                if not text: raise ValueError("Empty response")
                return text
            except Exception as e:
                self.logger.warning(f"Attempt {attempt+1} failed: {e}")
                if attempt == self.max_retries - 1: raise
                time.sleep(2)
