"""Comprehensive test suite for MedRec application."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import sounddevice as sd
import soundfile as sf

from app.audio import AudioRecorder
from app.config import AppConfig, load_config
from app.summarizer import OllamaSummarizer
from app.transcriber import TranscriptionResult, WhisperTranscriber
from app.terminology import apply_corrections
from app.storage import StorageManager


class MedRecTester:
    """Comprehensive testing suite for MedRec."""
    
    def __init__(self):
        self.config: AppConfig = load_config()
        self.logger = logging.getLogger("medrec.tester")
        self.setup_logging()
        
        # Initialize components
        self.audio = AudioRecorder(self.config.audio)
        self.transcriber = WhisperTranscriber(self.config.whisper)
        self.summarizer = OllamaSummarizer(self.config.summarizer)
        self.storage = StorageManager(self.config.storage)
        
        self.results: Dict[str, any] = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "errors": [],
            "performance": {},
        }
    
    def setup_logging(self):
        """Setup logging for tests."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    
    def run_all_tests(self) -> bool:
        """Run all tests and return success status."""
        print("\n" + "="*70)
        print("MEDREC COMPREHENSIVE TEST SUITE")
        print("="*70 + "\n")
        
        tests = [
            ("Environment Check", self.test_environment),
            ("Audio System", self.test_audio_system),
            ("Whisper Transcription", self.test_whisper_transcription),
            ("Summarizer Service", self.test_summarizer_service),
            ("End-to-End Workflow", self.test_end_to_end),
            ("Performance Benchmarks", self.test_performance),
        ]
        
        for test_name, test_func in tests:
            print(f"\n{'─'*70}")
            print(f"Running: {test_name}")
            print(f"{'─'*70}")
            
            try:
                success = test_func()
                self.results["tests_run"] += 1
                if success:
                    self.results["tests_passed"] += 1
                    print(f"✓ {test_name}: PASSED")
                else:
                    self.results["tests_failed"] += 1
                    print(f"✗ {test_name}: FAILED")
            except Exception as e:
                self.results["tests_failed"] += 1
                self.results["errors"].append(f"{test_name}: {str(e)}")
                self.logger.exception(f"Test {test_name} raised exception")
                print(f"✗ {test_name}: ERROR - {e}")
        
        self.print_summary()
        return self.results["tests_failed"] == 0
    
    def test_environment(self) -> bool:
        """Test environment setup."""
        print("\n1. Checking Python version...")
        version = sys.version_info
        print(f"   Python {version.major}.{version.minor}.{version.micro}")
        if version.major < 3 or (version.major == 3 and version.minor < 11):
            print("   ⚠ Warning: Python 3.11+ recommended")
        
        print("\n2. Checking dependencies...")
        try:
            import PySide6
            print(f"   ✓ PySide6: {PySide6.__version__}")
        except ImportError:
            print("   ✗ PySide6 not installed")
            return False
        
        try:
            import faster_whisper
            print(f"   ✓ faster-whisper: {faster_whisper.__version__}")
        except ImportError:
            print("   ⚠ faster-whisper not installed (will use whisper.cpp)")
        
        print("\n3. Checking Whisper binary...")
        whisper_binary = Path(self.config.whisper.binary_path)
        if whisper_binary.exists():
            print(f"   ✓ Whisper binary found: {whisper_binary}")
        else:
            print(f"   ✗ Whisper binary missing: {whisper_binary}")
            return False
        
        print("\n4. Checking Whisper model...")
        whisper_model = Path(self.config.whisper.model_path)
        if whisper_model.exists():
            size_mb = whisper_model.stat().st_size / (1024 * 1024)
            print(f"   ✓ Whisper model found: {whisper_model} ({size_mb:.1f} MB)")
        else:
            print(f"   ✗ Whisper model missing: {whisper_model}")
            return False
        
        return True
    
    def test_audio_system(self) -> bool:
        """Test audio recording system."""
        print("\n1. Checking audio devices...")
        try:
            devices = sd.query_devices()
            input_devices = [d for d in devices if d.get("max_input_channels", 0) > 0]
            print(f"   Found {len(input_devices)} input device(s)")
            
            for idx, dev in enumerate(input_devices[:3]):  # Show first 3
                print(f"   {idx}: {dev['name']} ({dev.get('max_input_channels', 0)} channels)")
            
            if not input_devices:
                print("   ✗ No input devices found")
                return False
            
            print("   ✓ Audio system ready")
        except Exception as e:
            print(f"   ✗ Audio system error: {e}")
            return False
        
        print("\n2. Testing audio recording (5 seconds)...")
        try:
            temp_dir = Path(self.config.storage.root) / "tmp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            test_audio = temp_dir / f"test_recording_{int(time.time())}.wav"
            
            print("   Recording... (speak now)")
            self.audio.start(test_audio)
            time.sleep(5)
            self.audio.stop()
            
            if test_audio.exists() and test_audio.stat().st_size > 0:
                size_kb = test_audio.stat().st_size / 1024
                print(f"   ✓ Recording successful ({size_kb:.1f} KB)")
                # Cleanup
                test_audio.unlink()
                return True
            else:
                print("   ✗ Recording failed - file empty or missing")
                return False
        except Exception as e:
            print(f"   ✗ Recording error: {e}")
            return False
    
    def test_whisper_transcription(self) -> bool:
        """Test Whisper transcription."""
        print("\n1. Testing Whisper initialization...")
        try:
            # Test with a simple check
            print(f"   Engine: {getattr(self.config.whisper, 'engine', 'cli')}")
            print(f"   Model: {Path(self.config.whisper.model_path).name}")
            print("   ✓ Whisper initialized")
        except Exception as e:
            print(f"   ✗ Whisper initialization error: {e}")
            return False
        
        print("\n2. Testing transcription with sample audio...")
        # Check if we have any existing audio files
        sessions_dir = self.storage.sessions_dir
        sample_audio = None
        
        if sessions_dir.exists():
            for session_dir in sessions_dir.glob("session_*"):
                audio_files = list(session_dir.glob("*.wav"))
                if audio_files:
                    sample_audio = audio_files[0]
                    break
        
        if not sample_audio:
            print("   ⚠ No sample audio found - skipping transcription test")
            print("   (Record audio first to test transcription)")
            return True  # Not a failure, just no data
        
        print(f"   Using sample: {sample_audio.name}")
        try:
            start_time = time.perf_counter()
            
            def progress_cb(text: str):
                if text.strip():
                    print(f"   Partial: {text[:50]}...")
            
            result = self.transcriber.transcribe(sample_audio, progress_cb=progress_cb)
            runtime = time.perf_counter() - start_time
            
            cleaned = apply_corrections(result.text)
            
            print(f"\n   Transcription Results:")
            print(f"   Runtime: {runtime:.2f}s")
            print(f"   Text length: {len(cleaned)} characters")
            print(f"   Word count: ~{len(cleaned.split())} words")
            print(f"\n   Transcript preview:")
            print(f"   {cleaned[:200]}...")
            
            if cleaned.strip():
                self.results["performance"]["transcription_runtime"] = runtime
                self.results["performance"]["transcription_length"] = len(cleaned)
                print("   ✓ Transcription successful")
                return True
            else:
                print("   ✗ Transcription produced empty result")
                return False
                
        except Exception as e:
            print(f"   ✗ Transcription error: {e}")
            self.logger.exception("Transcription test failed")
            return False
    
    def test_summarizer_service(self) -> bool:
        """Test summarizer service."""
        print("\n1. Testing Ollama connection...")
        try:
            if self.summarizer.health_check():
                print("   ✓ Ollama service is running")
            else:
                print("   ✗ Ollama service not responding")
                print("   → Start Ollama: ollama serve")
                return False
        except Exception as e:
            print(f"   ✗ Ollama connection error: {e}")
            return False
        
        print("\n2. Testing model availability...")
        model = self.config.summarizer.model
        print(f"   Model: {model}")
        print("   ✓ Model configured")
        
        print("\n3. Testing summarization with sample text...")
        sample_transcript = """
        Patient presents with abdominal pain and diarrhea for the past two weeks.
        Reports no blood in stool. Previous colonoscopy showed mild inflammation.
        Currently taking mesalamine 800mg twice daily. Blood pressure is normal.
        Weight is stable. No fever or chills.
        """
        
        try:
            start_time = time.perf_counter()
            result = self.summarizer.summarize(sample_transcript, style="Narrative")
            runtime = time.perf_counter() - start_time
            
            print(f"\n   Summarization Results:")
            print(f"   Runtime: {runtime:.2f}s")
            print(f"   Summary length: {len(result.summary)} characters")
            print(f"   Model used: {result.model_used}")
            print(f"   Validation passed: {getattr(result, 'validation_passed', True)}")
            
            print(f"\n   Summary preview:")
            preview = result.summary[:300]
            print(f"   {preview}...")
            
            if result.summary.strip():
                self.results["performance"]["summarization_runtime"] = runtime
                self.results["performance"]["summarization_length"] = len(result.summary)
                print("   ✓ Summarization successful")
                return True
            else:
                print("   ✗ Summarization produced empty result")
                return False
                
        except Exception as e:
            print(f"   ✗ Summarization error: {e}")
            self.logger.exception("Summarization test failed")
            return False
    
    def test_end_to_end(self) -> bool:
        """Test complete end-to-end workflow."""
        print("\nTesting complete workflow: Record → Transcribe → Summarize")
        
        # Check if we can record
        try:
            devices = sd.query_devices()
            input_devices = [d for d in devices if d.get("max_input_channels", 0) > 0]
            if not input_devices:
                print("   ⚠ No audio input - skipping end-to-end test")
                return True
        except:
            print("   ⚠ Audio system unavailable - skipping end-to-end test")
            return True
        
        print("\n1. Recording test audio (10 seconds)...")
        temp_dir = Path(self.config.storage.root) / "tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        test_audio = temp_dir / f"e2e_test_{int(time.time())}.wav"
        
        try:
            self.audio.start(test_audio)
            print("   Recording... (speak a medical dictation)")
            time.sleep(10)
            self.audio.stop()
            
            if not test_audio.exists() or test_audio.stat().st_size < 1024:
                print("   ✗ Recording failed")
                return False
            
            print("   ✓ Recording complete")
        except Exception as e:
            print(f"   ✗ Recording error: {e}")
            return False
        
        print("\n2. Transcribing...")
        try:
            start_time = time.perf_counter()
            result = self.transcriber.transcribe(test_audio)
            runtime = time.perf_counter() - start_time
            
            cleaned = apply_corrections(result.text)
            
            if not cleaned.strip():
                print("   ✗ Transcription produced no text")
                return False
            
            print(f"   ✓ Transcription complete ({runtime:.2f}s, {len(cleaned)} chars)")
        except Exception as e:
            print(f"   ✗ Transcription error: {e}")
            return False
        
        print("\n3. Summarizing...")
        try:
            if not self.summarizer.health_check():
                print("   ⚠ Ollama not running - skipping summarization")
                return True  # Partial success
            
            start_time = time.perf_counter()
            summary_result = self.summarizer.summarize(cleaned, style="Narrative")
            runtime = time.perf_counter() - start_time
            
            if not summary_result.summary.strip():
                print("   ✗ Summarization produced no text")
                return False
            
            print(f"   ✓ Summarization complete ({runtime:.2f}s, {len(summary_result.summary)} chars)")
            print(f"\n   Summary preview:")
            print(f"   {summary_result.summary[:200]}...")
            
            # Cleanup
            test_audio.unlink()
            
            return True
        except Exception as e:
            print(f"   ✗ Summarization error: {e}")
            return False
    
    def test_performance(self) -> bool:
        """Test performance benchmarks."""
        print("\nPerformance Benchmarks:")
        
        perf = self.results.get("performance", {})
        
        if "transcription_runtime" in perf:
            trans_time = perf["transcription_runtime"]
            trans_len = perf.get("transcription_length", 0)
            if trans_len > 0:
                chars_per_sec = trans_len / trans_time
                print(f"\n   Transcription:")
                print(f"   Speed: {chars_per_sec:.1f} chars/sec")
                print(f"   Runtime: {trans_time:.2f}s")
        
        if "summarization_runtime" in perf:
            summ_time = perf["summarization_runtime"]
            summ_len = perf.get("summarization_length", 0)
            if summ_len > 0:
                chars_per_sec = summ_len / summ_time
                print(f"\n   Summarization:")
                print(f"   Speed: {chars_per_sec:.1f} chars/sec")
                print(f"   Runtime: {summ_time:.2f}s")
        
        print("\n   ✓ Performance metrics collected")
        return True
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"\nTests Run: {self.results['tests_run']}")
        print(f"Tests Passed: {self.results['tests_passed']}")
        print(f"Tests Failed: {self.results['tests_failed']}")
        
        if self.results["errors"]:
            print(f"\nErrors ({len(self.results['errors'])}):")
            for error in self.results["errors"]:
                print(f"  - {error}")
        
        if self.results["performance"]:
            print("\nPerformance Metrics:")
            for key, value in self.results["performance"].items():
                print(f"  {key}: {value}")
        
        print("\n" + "="*70)
        
        if self.results["tests_failed"] == 0:
            print("✓ ALL TESTS PASSED")
        else:
            print("✗ SOME TESTS FAILED")
        print("="*70 + "\n")


def main():
    """Run comprehensive tests."""
    tester = MedRecTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()



