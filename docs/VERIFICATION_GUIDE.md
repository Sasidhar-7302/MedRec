# MedRec Verification Guide

This guide explains how to verify the installation, functionality, and accuracy of the MedRec application.

## 1. Quick Verification

To verify that the application core components are working correctly:

```bash
python -m tests.test_quick
```

This runs a minimal test suite that checks:
- Environment setup
- Audio system availability
- Whisper model loading
- Basic configuration

## 2. Comprehensive Testing

To run the full test suite (recommended before major updates):

```bash
python -m tests.test_comprehensive
```

This tests everything including:
- Real-time audio recording
- Whisper transcription
- Ollama summarization
- Storage system
- End-to-end workflow

## 3. Accuracy Analysis

To measure the accuracy of the summarization and transcription models:

```bash
python -m scripts.verify_accuracy
```

This script:
- Generates summaries for a validation dataset
- Compares results against ground truth
- Calculates accuracy scores for HPI, Assessment, and Plan sections
- Saves a report to `VERIFICATION_REPORT.md`

## 4. Manual Testing

For manual verification steps, see the application UI. The core workflow is:
1. Speak a dictation into the microphone.
2. Watch the live transcription.
3. Click "End & Summarize".
4. Verify the structured note appears in the right panel.
