# Quick Fix Guide for MedRec Issues

## Immediate Fixes (No Fine-Tuning Required)

### Issue 1: Summarization Producing Wrong Format

**Problem**: Summary is generating dialogue instead of structured clinical notes.

**Quick Fix**:
1. Check if Ollama is running: `ollama serve`
2. Verify model name in `config.json`:
   ```json
   {
     "summarizer": {
       "model": "medllama2:7b-instruct-q4_K_M"
     }
   }
   ```
3. Pull correct model:
   ```bash
   ollama pull medllama2:7b-instruct-q4_K_M
   ```
4. Test with simple prompt to verify model behavior

**If still not working**:
- The prompt templates have been updated with better instructions
- Restart Ollama service
- Clear any cached responses

---

### Issue 2: Slow Transcription

**Problem**: Transcription taking 70+ seconds for short audio.

**Quick Fixes**:

1. **Use Faster-Whisper (if available)**:
   ```json
   {
     "whisper": {
       "engine": "faster",
       "faster_model": "medium.en",
       "device": "cpu",
       "compute_type": "int8"
     }
   }
   ```

2. **Optimize Whisper.cpp settings**:
   ```json
   {
     "whisper": {
       "threads": 8,  // Match your CPU cores
       "beam_size": 3,  // Reduce from 5 for speed
       "temperature": 0.0
     }
   }
   ```

3. **Use smaller audio chunks**:
   - Record shorter segments
   - Process in batches

---

### Issue 3: Medical Terminology Errors

**Quick Fix**: Update terminology corrections

Edit `app/terminology.py` and add corrections:
```python
CORRECTIONS: List[Tuple[str, str]] = [
    # Add your specific corrections
    (r"\bpankal it is\b", "pancolitis"),
    (r"\bcoloscopy\b", "colonoscopy"),
    (r"\bburbs\b", "biopsies"),
    # ... add more
]
```

---

## Testing After Fixes

Run the test suite:
```bash
python test_application.py
```

Check specific components:
```bash
# Test transcription only
python -c "from app.transcriber import WhisperTranscriber; from app.config import load_config; t = WhisperTranscriber(load_config().whisper); print('OK')"

# Test summarizer only
python -c "from app.summarizer import OllamaSummarizer; from app.config import load_config; s = OllamaSummarizer(load_config().summarizer); print('OK' if s.health_check() else 'FAIL')"
```

---

## Next Steps for Fine-Tuning

See `GOOGLE_COLAB_FINETUNING_GUIDE.md` for:
- Complete fine-tuning instructions
- Data preparation
- Model training
- Deployment

---

*Last Updated: 2025-12-10*



