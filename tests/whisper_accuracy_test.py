"""
Whisper Transcription Accuracy Testing
=======================================
Generates synthetic audio from GI dialogues using TTS and measures Word Error Rate (WER).
Focuses on GI-specific terminology accuracy.
"""

import json
import os
import re
import tempfile
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple

# Set CUDA library path before imports
os.environ["PATH"] = str(Path(__file__).parent / "cuda_libs") + os.pathsep + os.environ.get("PATH", "")

from app.config import AppConfig
from app.transcriber import WhisperTranscriber
from app.gi_terms import load_gi_terms


@dataclass
class TranscriptionTestResult:
    """Result of a single transcription test."""
    original_text: str
    transcribed_text: str
    wer: float
    gi_wer: float
    time_s: float
    gi_terms_found: int
    gi_terms_total: int


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (WER) between reference and hypothesis."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    
    distance = levenshtein_distance(ref_words, hyp_words)
    return distance / len(ref_words)


def extract_gi_terms(text: str, gi_vocabulary: List[str]) -> List[str]:
    """Extract GI terms found in text."""
    text_lower = text.lower()
    found_terms = []
    for term in gi_vocabulary:
        if term.lower() in text_lower:
            found_terms.append(term)
    return found_terms


def calculate_gi_wer(reference: str, hypothesis: str, gi_vocabulary: List[str]) -> Tuple[float, int, int]:
    """Calculate GI-specific Word Error Rate."""
    ref_gi_terms = set(extract_gi_terms(reference, gi_vocabulary))
    hyp_gi_terms = set(extract_gi_terms(hypothesis, gi_vocabulary))
    
    if len(ref_gi_terms) == 0:
        return 0.0, 0, 0
    
    # Count correctly transcribed GI terms
    correct = len(ref_gi_terms.intersection(hyp_gi_terms))
    total = len(ref_gi_terms)
    
    gi_wer = 1.0 - (correct / total)
    return gi_wer, correct, total


def generate_audio_tts(text: str, output_path: Path) -> bool:
    """Generate audio from text using TTS."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Speaking rate
        engine.save_to_file(text, str(output_path))
        engine.runAndWait()
        return output_path.exists()
    except ImportError:
        # Fallback: try edge-tts for better quality
        try:
            import asyncio
            import edge_tts
            
            async def generate():
                communicate = edge_tts.Communicate(text, "en-US-GuyNeural")
                await communicate.save(str(output_path))
            
            asyncio.run(generate())
            return output_path.exists()
        except ImportError:
            print("WARNING: No TTS engine available. Install pyttsx3 or edge-tts.")
            return False


# Sample GI dialogues for testing
TEST_DIALOGUES = [
    {
        "id": "gi_001",
        "text": "Good morning. I've been having burning epigastric pain for about two months now. It's worse at night and when I'm lying flat. I've also noticed some regurgitation and occasional dysphagia.",
        "expected_gi_terms": ["epigastric", "regurgitation", "dysphagia"],
    },
    {
        "id": "gi_002",
        "text": "My assessment is ulcerative colitis with a mild to moderate flare. I recommend continuing vedolizumab and adding budesonide for eight weeks. We'll repeat your CBC, CMP, and CRP before your next visit.",
        "expected_gi_terms": ["ulcerative colitis", "vedolizumab", "budesonide", "CBC", "CMP", "CRP"],
    },
    {
        "id": "gi_003",
        "text": "I've been experiencing chronic diarrhea with mucus, about five to six times daily. I also have abdominal pain, fatigue, and some joint pain. My Crohn's disease was diagnosed two years ago.",
        "expected_gi_terms": ["diarrhea", "Crohn's disease", "abdominal"],
    },
    {
        "id": "gi_004",
        "text": "Let's schedule an EGD to evaluate your Barrett's esophagus. We should also order a colonoscopy for surveillance given your history of adenomatous polyps. Continue your omeprazole 20mg twice daily.",
        "expected_gi_terms": ["EGD", "Barrett's esophagus", "colonoscopy", "adenomatous polyp", "omeprazole"],
    },
    {
        "id": "gi_005",
        "text": "The patient presents with episodic right upper quadrant pain after fatty meals for about six months. Labs show elevated ALT and AST. I'm concerned about cholelithiasis or cholecystitis.",
        "expected_gi_terms": ["ALT", "AST", "cholelithiasis", "cholecystitis"],
    },
    {
        "id": "gi_006",
        "text": "We need to escalate to ustekinumab given the loss of response to infliximab. Order an MR enterography to assess the extent of small bowel involvement. Check therapeutic drug monitoring and antibody levels.",
        "expected_gi_terms": ["ustekinumab", "infliximab", "MR enterography", "small bowel"],
    },
    {
        "id": "gi_007",
        "text": "Start a low FODMAP diet and order celiac serologies. Also test for H. pylori. The functional dyspepsia should improve with dietary modifications and we'll add a proton pump inhibitor if needed.",
        "expected_gi_terms": ["FODMAP", "celiac", "H. pylori", "functional dyspepsia", "proton pump inhibitor"],
    },
    {
        "id": "gi_008",
        "text": "The capsule endoscopy showed multiple ulcerations in the jejunum and ileum consistent with Crohn's disease. Fecal calprotectin is elevated at 850. I recommend starting Stelara.",
        "expected_gi_terms": ["capsule endoscopy", "jejunum", "ileum", "Crohn's disease", "calprotectin", "Stelara"],
    },
]


def run_transcription_tests():
    print("=" * 70)
    print("GI SCRIBE - WHISPER TRANSCRIPTION ACCURACY TEST")
    print("=" * 70)
    print(f"Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load configuration
    config = AppConfig.load(Path("config.json"))
    transcriber = WhisperTranscriber(config.whisper)
    gi_vocabulary = load_gi_terms()

    print(f"GI Vocabulary Size: {len(gi_vocabulary)} terms")
    print(f"Test Cases: {len(TEST_DIALOGUES)}")
    print()

    results: List[TranscriptionTestResult] = []
    total_time = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, test in enumerate(TEST_DIALOGUES, 1):
            print(f"\n[Test {i}/{len(TEST_DIALOGUES)}] {test['id']}")
            print("-" * 50)

            # Generate audio
            audio_path = Path(tmpdir) / f"{test['id']}.mp3"
            print(f"Generating audio...")
            
            if not generate_audio_tts(test["text"], audio_path):
                print("  SKIPPED: Could not generate audio")
                continue

            # Transcribe
            print(f"Transcribing...")
            start = time.perf_counter()
            try:
                result = transcriber.transcribe(audio_path)
                elapsed = time.perf_counter() - start
                transcribed = result.text
                total_time += elapsed
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

            # Calculate metrics
            wer = calculate_wer(test["text"], transcribed)
            gi_wer, gi_found, gi_total = calculate_gi_wer(test["text"], transcribed, gi_vocabulary)

            test_result = TranscriptionTestResult(
                original_text=test["text"],
                transcribed_text=transcribed,
                wer=wer,
                gi_wer=gi_wer,
                time_s=elapsed,
                gi_terms_found=gi_found,
                gi_terms_total=gi_total,
            )
            results.append(test_result)

            # Print results
            print(f"  Time: {elapsed:.2f}s")
            print(f"  WER: {wer*100:.1f}%")
            print(f"  GI-WER: {gi_wer*100:.1f}% ({gi_found}/{gi_total} terms correct)")
            print(f"  Original: {test['text'][:80]}...")
            print(f"  Transcribed: {transcribed[:80]}...")

    # Summary Statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    if results:
        avg_wer = sum(r.wer for r in results) / len(results)
        avg_gi_wer = sum(r.gi_wer for r in results) / len(results)
        avg_time = sum(r.time_s for r in results) / len(results)
        total_gi_found = sum(r.gi_terms_found for r in results)
        total_gi = sum(r.gi_terms_total for r in results)

        print(f"Tests Completed: {len(results)}/{len(TEST_DIALOGUES)}")
        print()
        print("TRANSCRIPTION ACCURACY:")
        print(f"  Average WER: {avg_wer*100:.1f}%")
        print(f"  Average GI-WER: {avg_gi_wer*100:.1f}%")
        print(f"  GI Terms Accuracy: {(1-avg_gi_wer)*100:.1f}% ({total_gi_found}/{total_gi} terms)")
        print()
        print("PERFORMANCE:")
        print(f"  Average Time: {avg_time:.2f}s per clip")
        print(f"  Total Time: {total_time:.2f}s")
        print()
        
        # Overall score
        accuracy = (1 - avg_wer) * 100
        gi_accuracy = (1 - avg_gi_wer) * 100
        print("OVERALL ASSESSMENT:")
        if accuracy >= 95 and gi_accuracy >= 95:
            print(f"  Transcription: {accuracy:.0f}% - EXCELLENT ✓")
            print(f"  GI Terms: {gi_accuracy:.0f}% - EXCELLENT ✓")
        elif accuracy >= 90 and gi_accuracy >= 90:
            print(f"  Transcription: {accuracy:.0f}% - GOOD")
            print(f"  GI Terms: {gi_accuracy:.0f}% - GOOD")
        else:
            print(f"  Transcription: {accuracy:.0f}% - NEEDS IMPROVEMENT")
            print(f"  GI Terms: {gi_accuracy:.0f}% - NEEDS IMPROVEMENT")
    else:
        print("No tests completed! Check TTS installation.")
        print("Install with: pip install pyttsx3 or pip install edge-tts")

    print("\n" + "=" * 70)
    print(f"Test Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_transcription_tests()
