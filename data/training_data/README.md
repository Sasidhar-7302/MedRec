# Training Data for Fine-Tuning

**Generated:** December 17, 2025  
**Total Samples:** 2000 (~20 hours equivalent)  
**Status:** ✅ Ready for Google Colab Fine-Tuning

## Files Generated

### Full Dataset
- **`full_dataset.jsonl`** - Complete dataset with all fields (2000 entries)
  - Contains: dialogue, transcript, summary, HPI, findings, assessment, plan, speakers

### Whisper Fine-Tuning
- **`whisper_train.jsonl`** - Training set for Whisper (1800 entries)
- **`whisper_val.jsonl`** - Validation set for Whisper (200 entries)
  - Format: `{"id": "...", "text": "...", "speakers": [...], "source": "synthetic"}`

### MedLlama Fine-Tuning
- **`medllama_train.jsonl`** - Training set for HPI/Assessment extraction (1800 entries)
- **`medllama_val.jsonl`** - Validation set for HPI/Assessment extraction (200 entries)
  - Format: `{"instruction": "...", "input": "...", "output": "...", "hpi": "...", "assessment": "..."}`

### Summary
- **`summary.json`** - Dataset statistics and metadata

## Data Characteristics

### Speaker Differentiation
- ✅ All dialogues include speaker labels (Doctor/Patient)
- ✅ Proper speaker diarization format
- ✅ Ready for speaker token training

### HPI (History of Present Illness)
- ✅ Extracted from patient statements
- ✅ Includes: demographics, timeline, symptoms, associated symptoms, history
- ✅ Proper medical narrative format

### Assessment
- ✅ Structured format with numbered problems
- ✅ Includes severity and activity
- ✅ Links findings to diagnoses
- ✅ Clear clinical reasoning

### Medical Terminology
- ✅ GI-specific terminology
- ✅ Proper medical language
- ✅ Common conditions and treatments

## Usage in Google Colab

### For Whisper Fine-Tuning

1. Upload `whisper_train.jsonl` and `whisper_val.jsonl` to Google Drive
2. Follow [Google Colab Fine-Tuning Guide](../../docs/GOOGLE_COLAB_FINETUNING_GUIDE.md)
3. Use the manifests in Part 1 (Speaker Diarization) or Part 2 (Medical Terminology)

### For MedLlama Fine-Tuning

1. Upload `medllama_train.jsonl` and `medllama_val.jsonl` to Google Drive
2. Follow [Google Colab Fine-Tuning Guide](../../docs/GOOGLE_COLAB_FINETUNING_GUIDE.md)
3. Use the data in Part 3 (HPI/Assessment Extraction)

## Data Format Examples

### Whisper Format
```json
{
  "id": "train_00001",
  "text": "Good morning, thanks for coming in today. What brings you in?\nI've been having abdominal pain for three days now.",
  "text_labeled": "Doctor: Good morning, thanks for coming in today. What brings you in?\nPatient: I've been having abdominal pain for three days now.",
  "speakers": ["doctor", "patient", "doctor"],
  "source": "synthetic",
  "note": "Audio file needs to be generated using TTS. Use 'text' for realistic training, 'text_labeled' for supervision."
}
```

**Key Points:**
- `text`: **Unlabeled** (realistic conversation - no "Doctor:" or "Patient:" labels)
- `text_labeled`: **Labeled** (for training supervision)
- `speakers`: True speaker sequence array
- Model learns to identify speakers from audio patterns or context clues

### MedLlama Format
```json
{
  "instruction": "Extract HPI (History of Present Illness) and Assessment from this medical conversation transcript. Identify patient statements for HPI and doctor statements for Assessment.",
  "input": "Good morning, thanks for coming in today. What brings you in?\nI've been having abdominal pain for three days now.",
  "input_labeled": "Doctor: Good morning...\nPatient: I've been having...",
  "output": "HPI (History of Present Illness):\n54-year-old male working as teacher...\n\nAssessment:\n1. Ulcerative colitis, mild to moderate flare...",
  "hpi": "54-year-old male working as teacher presents with...",
  "assessment": "1. Ulcerative colitis, mild to moderate flare - mild to moderate severity, active activity...",
  "speakers": ["doctor", "patient", "doctor"]
}
```

**Key Points:**
- `input`: **Unlabeled** (realistic - model learns to identify speakers from context)
- `input_labeled`: **Labeled** (for reference/validation)
- `speakers`: True speaker sequence for validation
- Model learns to extract HPI from patient statements and Assessment from doctor statements

## Next Steps

1. **Review Data Quality:** Check a few samples to ensure format is correct
2. **Upload to Google Drive:** For Colab access
3. **Follow Fine-Tuning Guide:** Use [GOOGLE_COLAB_FINETUNING_GUIDE.md](../../docs/GOOGLE_COLAB_FINETUNING_GUIDE.md)
4. **Fine-Tune Models:** In Google Colab
5. **Deploy:** Integrate fine-tuned models into MedRec

## Notes

- **Audio Generation:** Whisper manifests note that audio files need to be generated using TTS
- **Synthetic Data:** All data is synthetically generated but follows realistic medical conversation patterns
- **Real Data:** For production, supplement with real medical dictations when available
- **Speaker Labels:** Training data includes both unlabeled (realistic) and labeled (reference) versions
- **Context Learning:** Models learn to identify speakers from context clues (medical terminology, question patterns, etc.)

## How Speaker Differentiation Works

**In Real Conversations:**
- Speakers don't say "Doctor:" or "Patient:"
- System uses **speaker diarization** (audio analysis) or **context clues** (text analysis) to identify speakers
- See [SPEAKER_DIARIZATION_GUIDE.md](../../docs/SPEAKER_DIARIZATION_GUIDE.md) for details

**Training Strategy:**
- Model sees **unlabeled conversations** (realistic)
- Uses **labeled versions** as supervision signal
- Learns to infer speakers from:
  - Medical terminology (doctor uses more)
  - Question patterns (doctor asks questions)
  - First-person statements (patient reports symptoms)
  - Treatment language (doctor discusses plans)

---

**For detailed fine-tuning instructions, see:** [docs/GOOGLE_COLAB_FINETUNING_GUIDE.md](../../docs/GOOGLE_COLAB_FINETUNING_GUIDE.md)

