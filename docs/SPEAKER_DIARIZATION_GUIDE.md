# Speaker Diarization Guide
**How the System Distinguishes Doctor from Patient in Real Conversations**

## The Problem

In real medical conversations, speakers **don't say "Doctor:" or "Patient:"** - those labels don't exist in actual speech. The system needs to identify who is speaking using other methods.

## Two Approaches

### 1. Audio-Based Speaker Diarization (Primary Method)

**How it works:**
- Analyzes audio characteristics (voice patterns, pitch, timbre)
- Identifies different speakers based on acoustic features
- Works even when speakers don't identify themselves

**Implementation:**
```python
# Using pyannote.audio
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipeline(audio_file)

# Output: Speaker segments with timestamps
# Speaker 0: [0.0s -> 2.5s] "Good morning, thanks for coming in..."
# Speaker 1: [2.5s -> 5.0s] "I've been having abdominal pain..."
```

**Advantages:**
- ✅ Works on any conversation
- ✅ Doesn't require text analysis
- ✅ Can identify multiple speakers
- ✅ Accurate (85-90% accuracy)

**Limitations:**
- Requires stereo audio or multiple microphones for best results
- May struggle with similar voices
- Needs audio preprocessing

### 2. Context-Based Identification (Fallback Method)

**How it works:**
- Analyzes text patterns and language
- Identifies speakers based on:
  - **Medical terminology** (doctors use more)
  - **Question patterns** (doctors ask questions)
  - **First-person statements** (patients report symptoms)
  - **Treatment language** (doctors discuss plans)

**Example Patterns:**

**Doctor Speech:**
- "What brings you in today?"
- "Any bleeding, fever, or weight loss?"
- "My assessment is..."
- "Let's review your labs..."
- Uses medical terms: "hematochezia", "pancolitis", "vedolizumab"

**Patient Speech:**
- "I've been having..."
- "It started..."
- "I have..."
- "What does that mean?"
- Reports symptoms in first person

**Implementation:**
```python
def identify_speaker_from_context(text_segment):
    # Doctor indicators
    doctor_patterns = [
        r"what brings you",
        r"my assessment",
        r"let's review",
        r"based on.*symptoms",
        r"hematochezia|pancolitis|vedolizumab",  # Medical terms
    ]
    
    # Patient indicators
    patient_patterns = [
        r"i've been having",
        r"it started",
        r"i have",
        r"what does that mean",
    ]
    
    # Check patterns
    doctor_score = sum(1 for pattern in doctor_patterns if re.search(pattern, text.lower()))
    patient_score = sum(1 for pattern in patient_patterns if re.search(pattern, text.lower()))
    
    if doctor_score > patient_score:
        return "doctor"
    elif patient_score > doctor_score:
        return "patient"
    else:
        return "unknown"
```

**Advantages:**
- ✅ Works with text-only transcripts
- ✅ No audio requirements
- ✅ Can be fine-tuned with training data

**Limitations:**
- Less accurate than audio-based (70-80%)
- May misidentify in ambiguous cases
- Requires good training data

## Training Data Strategy

### Why We Include Both Versions

**Unlabeled (Realistic):**
```
"Good morning, thanks for coming in today. What brings you in?"
"I've been having abdominal pain for three days now."
"Can you tell me more about when this started?"
```

**Labeled (For Training):**
```
"Doctor: Good morning, thanks for coming in today. What brings you in?"
"Patient: I've been having abdominal pain for three days now."
"Doctor: Can you tell me more about when this started?"
```

**Training Process:**
1. Model sees **unlabeled conversations** (realistic)
2. Uses **labeled versions** as supervision signal
3. Learns to infer speakers from:
   - Audio characteristics (if using diarization)
   - Context clues (if using text-based)
   - Or both (hybrid approach)

## Implementation in MedRec

### Current Approach

**For Transcription:**
1. Record audio (mono or stereo)
2. Transcribe with Whisper (no speaker labels)
3. Optionally run speaker diarization on audio
4. Match diarization results to transcript segments
5. Add speaker labels to transcript

**For Summarization:**
1. Receive transcript (with or without speaker labels)
2. If unlabeled, use context-based identification
3. Extract HPI from patient statements
4. Extract Assessment from doctor statements
5. Generate structured summary

### Recommended Workflow

**Best Practice:**
1. **Record in stereo** (doctor on one channel, patient on other)
   - Enables accurate audio-based diarization
   - Whisper.cpp supports stereo diarization

2. **Use pyannote.audio** for automatic diarization
   - More accurate than context-based
   - Works with mono audio too

3. **Fallback to context-based** if diarization fails
   - Use text patterns to identify speakers
   - Fine-tune with training data

## Fine-Tuning for Speaker Identification

### Whisper Fine-Tuning

**Goal:** Teach Whisper to output speaker tokens

**Training Data:**
- Audio files with multiple speakers
- Transcripts with speaker labels
- Model learns to associate audio patterns with speakers

**Expected Result:**
- Whisper outputs: `<|doctor|> Good morning...` `<|patient|> I've been having...`

### MedLlama Fine-Tuning

**Goal:** Extract HPI/Assessment even without explicit speaker labels

**Training Data:**
- Unlabeled conversations (realistic)
- HPI extracted from patient statements
- Assessment extracted from doctor statements

**Expected Result:**
- Model identifies patient statements for HPI
- Model identifies doctor statements for Assessment
- Works even without explicit labels

## Accuracy Expectations

| Method | Accuracy | Best For |
|--------|----------|----------|
| **Audio Diarization (Stereo)** | 90-95% | Best accuracy, requires stereo audio |
| **Audio Diarization (Mono)** | 85-90% | Good accuracy, works with mono |
| **Context-Based (Text)** | 70-80% | Fallback, text-only scenarios |
| **Hybrid (Audio + Text)** | 92-97% | Best overall, combines both |

## Practical Recommendations

### For Best Results

1. **Use stereo recording** when possible
   - Doctor and patient on separate channels
   - Enables accurate audio-based diarization

2. **Fine-tune Whisper** with speaker tokens
   - Model learns to identify speakers from audio
   - Outputs speaker labels automatically

3. **Fine-tune MedLlama** for context understanding
   - Learns to identify speakers from text patterns
   - Works as fallback when audio diarization unavailable

4. **Use hybrid approach**
   - Primary: Audio-based diarization
   - Fallback: Context-based identification
   - Best of both worlds

### For Current Setup

**If you have mono audio:**
- Use pyannote.audio for diarization
- Or use context-based identification
- Fine-tune models for better accuracy

**If you have stereo audio:**
- Use audio-based diarization (best accuracy)
- Whisper.cpp supports stereo diarization
- pyannote.audio works great with stereo

## Summary

**Key Points:**
1. Real conversations don't have speaker labels
2. System uses **audio diarization** (primary) or **context clues** (fallback)
3. Training data includes both unlabeled (realistic) and labeled (supervision) versions
4. Fine-tuning improves accuracy significantly
5. Hybrid approach (audio + text) gives best results

**Next Steps:**
- See [Google Colab Fine-Tuning Guide](GOOGLE_COLAB_FINETUNING_GUIDE.md) for training
- Implement pyannote.audio for audio-based diarization
- Fine-tune models for your specific use case

---

**Last Updated:** December 17, 2025


