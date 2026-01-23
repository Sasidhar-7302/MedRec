# Accuracy Improvement Guide
**Last Updated:** December 17, 2025

This guide provides comprehensive strategies for improving MedRec accuracy, especially for:
- Speaker differentiation (Doctor vs Patient)
- HPI (History of Present Illness) extraction
- Assessment accuracy
- Medical terminology recognition

---

## Table of Contents

1. [Current Accuracy Status](#current-accuracy-status)
2. [Improvement Strategies](#improvement-strategies)
3. [Fine-Tuning Workflow](#fine-tuning-workflow)
4. [Data Collection Best Practices](#data-collection-best-practices)
5. [Prompt Engineering](#prompt-engineering)
6. [Validation & Testing](#validation--testing)

---

## Current Accuracy Status

Based on comprehensive testing:

| Component | Current Accuracy | Target Accuracy |
|-----------|------------------|-----------------|
| **Transcription (Medical Terms)** | 80-85% | 95%+ |
| **Speaker Diarization** | Not implemented | 90%+ |
| **HPI Extraction** | 70-75% | 95%+ |
| **Assessment Accuracy** | 75-80% | 95%+ |
| **Overall Summary Quality** | Good | Excellent |

---

## Improvement Strategies

### 1. Speaker Differentiation

**Current State:** Transcripts don't distinguish between doctor and patient speech.

**Solution:**
1. **Fine-tune Whisper with speaker tokens** (see Google Colab guide)
2. **Use pyannote.audio for diarization** (automatic speaker separation)
3. **Update prompts** to handle speaker-labeled transcripts

**Implementation:**
```python
# Add speaker tokens to Whisper vocabulary
speaker_tokens = ["<|doctor|>", "<|patient|>", "<|nurse|>"]

# Fine-tune with speaker-labeled data
# Format: "Doctor: What brings you in? Patient: Abdominal pain..."
```

**Expected Improvement:** 85-90% speaker identification accuracy

---

### 2. HPI Extraction

**Current State:** HPI extraction is ~70-75% accurate, sometimes missing key details.

**Solution:**
1. **Enhanced prompt templates** (already updated)
2. **Fine-tune MedLlama** specifically for HPI extraction
3. **Few-shot examples** with proper HPI format

**Key Improvements:**
- Extract from patient statements only
- Include timeline (onset, duration)
- Character of symptoms
- Associated symptoms
- What makes better/worse

**Expected Improvement:** 90-95% HPI accuracy

---

### 3. Assessment Accuracy

**Current State:** Assessment sometimes misses diagnoses or severity.

**Solution:**
1. **Structured assessment format** (numbered problems)
2. **Link findings to diagnoses** explicitly
3. **Include severity/activity** (mild/moderate/severe)

**Key Improvements:**
- Number multiple problems (1., 2., etc.)
- Include supporting evidence
- Specify severity/activity
- Link to HPI and findings

**Expected Improvement:** 92-97% assessment accuracy

---

## Fine-Tuning Workflow

### Step 1: Data Collection

**Minimum Requirements:**
- 20-50 hours of medical dictation audio
- Speaker-labeled transcripts
- Structured summaries with HPI and Assessment

**Data Format:**
```json
{
  "audio": "path/to/audio.wav",
  "transcript": "Doctor: What brings you in? Patient: Abdominal pain...",
  "speakers": ["doctor", "patient", "doctor"],
  "hpi": "50-year-old male with 3-day history of...",
  "assessment": "1. Inflammatory bowel disease, mild activity...",
  "findings": "...",
  "plan": "..."
}
```

### Step 2: Data Preparation

1. **Speaker Diarization:**
   ```python
   # Use pyannote.audio
   from pyannote.audio import Pipeline
   pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
   ```

2. **Transcript Alignment:**
   - Align Whisper transcripts with speaker segments
   - Label each segment with speaker

3. **Summary Extraction:**
   - Extract HPI from patient statements
   - Extract Assessment from doctor's clinical reasoning

### Step 3: Fine-Tuning

**Whisper Fine-Tuning:**
- Focus on medical terminology
- Add speaker tokens
- Train on speaker-labeled data

**MedLlama Fine-Tuning:**
- Focus on HPI/Assessment extraction
- Use structured examples
- Train on conversation → summary pairs

### Step 4: Validation

Test on held-out data:
- Calculate WER for transcription
- Calculate F1 for HPI extraction
- Calculate accuracy for Assessment
- Measure speaker diarization accuracy

---

## Data Collection Best Practices

### Audio Quality

**Requirements:**
- Sample rate: 16kHz minimum
- Format: WAV (uncompressed) or FLAC
- Duration: 30 seconds to 5 minutes per clip
- Background noise: Minimal

**Recording Tips:**
- Use good microphone
- Quiet environment
- Clear speech
- Consistent volume

### Transcript Quality

**Requirements:**
- Word-level accuracy
- Speaker labels (Doctor/Patient)
- Timestamps for segments
- Medical terminology correct

**Transcription Tips:**
- Review and correct automatically generated transcripts
- Verify medical terminology
- Check speaker labels
- Validate timestamps

### Summary Quality

**Requirements:**
- Structured format (HPI, Findings, Assessment, Plan)
- Complete information
- Accurate medical terminology
- Clear and concise

**Summary Tips:**
- Use consistent format
- Include all relevant information
- Verify medical accuracy
- Review with clinicians

---

## Prompt Engineering

### Enhanced HPI Extraction

**Key Elements:**
1. **Patient Demographics:** Age, gender (if mentioned)
2. **Chief Complaint:** Primary reason for visit
3. **Timeline:** When symptoms started, duration
4. **Character:** Description of symptoms
5. **Associated Symptoms:** Related symptoms
6. **Modifying Factors:** What makes better/worse
7. **Relevant History:** Past medical history related to current complaint

**Example Prompt:**
```
Extract HPI from patient's statements:
- When did symptoms start?
- How long have they lasted?
- What do they feel like?
- What makes them better/worse?
- Any associated symptoms?
- Relevant past history?
```

### Enhanced Assessment

**Key Elements:**
1. **Numbered Problems:** List each diagnosis separately
2. **Supporting Evidence:** Link to findings
3. **Severity/Activity:** Mild/moderate/severe, active/remission
4. **Differential Diagnoses:** If applicable

**Example Format:**
```
Assessment:
1. Inflammatory bowel disease, likely ulcerative colitis, mild activity
   - Supporting: HPI of abdominal pain and hematochezia, previous colonoscopy showing inflammation
2. Abdominal pain, acute, likely related to active inflammation
```

---

## Validation & Testing

### Metrics

**Transcription:**
- Word Error Rate (WER)
- Medical Terminology Accuracy
- Speaker Diarization Accuracy

**Summarization:**
- HPI Extraction F1 Score
- Assessment Accuracy
- Completeness Score

### Testing Protocol

1. **Hold-out Test Set:** 10-20% of data
2. **Blind Review:** Clinicians review summaries
3. **Error Analysis:** Identify common mistakes
4. **Iterative Improvement:** Fine-tune based on errors

### Success Criteria

**Target Metrics:**
- Transcription WER: <5% for medical terms
- Speaker Diarization: >90% accuracy
- HPI Extraction: >95% F1 score
- Assessment Accuracy: >95%
- Overall Summary Quality: >90% clinician approval

---

## Quick Start: Improving Accuracy

### Immediate Actions (No Fine-Tuning)

1. **Update Prompt Templates:**
   - Use enhanced prompts (already updated)
   - Add more few-shot examples
   - Emphasize HPI and Assessment

2. **Expand Medical Vocabulary:**
   - Add terms to `data/gi_terms.txt`
   - Update terminology corrections
   - Add common misspellings

3. **Improve Data Quality:**
   - Review and correct transcripts
   - Validate summaries
   - Collect more diverse examples

### Short-Term (1-2 Weeks)

1. **Collect Training Data:**
   - Record 10-20 hours of dictations
   - Label speakers
   - Create structured summaries

2. **Fine-Tune Models:**
   - Use Google Colab guide
   - Fine-tune Whisper for medical terms
   - Fine-tune MedLlama for HPI/Assessment

3. **Validate Improvements:**
   - Test on held-out data
   - Measure accuracy improvements
   - Iterate based on results

### Long-Term (1-3 Months)

1. **Production Data Collection:**
   - Collect 50+ hours of real dictations
   - Continuous improvement
   - Regular fine-tuning cycles

2. **Advanced Features:**
   - Real-time transcription
   - Multi-speaker support
   - Custom model per doctor

---

## Resources

- [Google Colab Fine-Tuning Guide](GOOGLE_COLAB_FINETUNING_GUIDE.md)
- [Fine-Tuning Playbook](FINETUNING_PLAYBOOK.md)
- [Data and Training Guide](DATA_AND_TRAINING.md)

---

**Last Updated:** December 17, 2025


