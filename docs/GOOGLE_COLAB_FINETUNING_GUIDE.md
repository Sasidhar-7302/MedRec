# Google Colab Fine-Tuning Guide for Maximum Accuracy
**Last Updated:** December 17, 2025

This guide provides step-by-step instructions for fine-tuning Whisper and MedLlama models in Google Colab to achieve maximum accuracy, especially for:
- **Speaker Differentiation** (Doctor vs Patient)
- **HPI (History of Present Illness) Extraction**
- **Assessment Accuracy**
- **Medical Terminology Recognition**

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Part 1: Fine-Tuning Whisper for Speaker Diarization](#part-1-whisper-speaker-diarization)
3. [Part 2: Fine-Tuning Whisper for Medical Terminology](#part-2-whisper-medical-terminology)
4. [Part 3: Fine-Tuning MedLlama for HPI/Assessment](#part-3-medllama-hpi-assessment)
5. [Data Preparation](#data-preparation)
6. [Deployment](#deployment)
7. [Accuracy Improvements](#accuracy-improvements)

---

## Prerequisites

### Required Accounts
- Google Account (for Colab)
- Hugging Face account (for model hosting, optional)

### Required Data
- Medical dictation audio files (WAV/MP3)
- Corresponding transcripts with speaker labels
- Structured summaries with HPI and Assessment sections

### Storage
- Google Drive (for data storage)
- ~20-50 hours of audio for production-grade accuracy

---

## Part 1: Fine-Tuning Whisper for Speaker Diarization

### Step 1: Setup Google Colab Environment

```python
# Run this in Colab
!pip install -q transformers datasets accelerate peft bitsandbytes
!pip install -q torch torchvision torchaudio
!pip install -q librosa soundfile
!pip install -q jiwer  # For WER calculation
```

### Step 2: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Set your data path
DATA_PATH = '/content/drive/MyDrive/MedRec_Training'
```

### Step 3: Prepare Speaker-Diarized Dataset

Create a JSONL file with speaker labels:

```python
# Format: data/speaker_diarized_train.jsonl
{
  "audio": "/path/to/audio.wav",
  "text": "Doctor: Patient presents with abdominal pain. Patient: Yes, it started three days ago. Doctor: Any blood in stool?",
  "speakers": ["doctor", "patient", "doctor"],
  "segments": [
    {"start": 0.0, "end": 2.5, "speaker": "doctor", "text": "Patient presents with abdominal pain."},
    {"start": 2.5, "end": 5.0, "speaker": "patient", "text": "Yes, it started three days ago."},
    {"start": 5.0, "end": 7.0, "speaker": "doctor", "text": "Any blood in stool?"}
  ]
}
```

### Step 4: Fine-Tune Whisper with Speaker Tokens

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio
import torch
from peft import LoraConfig, get_peft_model, TaskType

# Load base model
model_name = "openai/whisper-medium.en"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Add speaker tokens to vocabulary
speaker_tokens = ["<|doctor|>", "<|patient|>", "<|nurse|>"]
processor.tokenizer.add_tokens(speaker_tokens)
model.resize_token_embeddings(len(processor.tokenizer))

# Configure LoRA for efficient fine-tuning
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION,
)

model = get_peft_model(model, lora_config)

# Load dataset
def prepare_dataset(batch):
    audio = batch["audio"]
    # Add speaker tokens to transcript
    text = batch["text"]  # Already includes speaker labels
    batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = processor.tokenizer(text).input_ids
    return batch

dataset = load_dataset("json", data_files=f"{DATA_PATH}/speaker_diarized_train.jsonl", split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)

# Training setup
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./whisper-speaker-diarized",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    warmup_steps=500,
    max_steps=5000,
    gradient_checkpointing=True,
    fp16=True,
    logging_steps=25,
    save_steps=500,
    eval_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor.feature_extractor,
)

# Train
trainer.train()

# Save model
model.save_pretrained("./whisper-speaker-diarized-final")
processor.save_pretrained("./whisper-speaker-diarized-final")
```

### Step 5: Convert to CTranslate2 Format (for faster-whisper)

```python
!pip install -q ctranslate2

import ctranslate2

# Convert to CTranslate2 format
ctranslate2.converters.TransformersConverter(
    model_name_or_path="./whisper-speaker-diarized-final",
    output_dir="./whisper-speaker-ct2",
    quantization="int8",
).convert()
```

---

## Part 2: Fine-Tuning Whisper for Medical Terminology

### Step 1: Prepare Medical Vocabulary Dataset

```python
# Create medical terminology training data
# Format: data/medical_terminology_train.jsonl
{
  "audio": "/path/to/audio.wav",
  "text": "Patient has pancolitis with hematochezia. Colonoscopy showed sigmoid inflammation. Currently on vedolizumab.",
  "medical_terms": ["pancolitis", "hematochezia", "sigmoid", "vedolizumab"],
  "corrected_text": "Patient has pancolitis with hematochezia. Colonoscopy showed sigmoid inflammation. Currently on vedolizumab."
}
```

### Step 2: Fine-Tune with Medical Vocabulary Bias

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model

# Load base model
model_name = "openai/whisper-medium.en"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Load medical vocabulary
with open(f"{DATA_PATH}/gi_terms.txt", "r") as f:
    medical_terms = [line.strip() for line in f if line.strip()]

# Add medical terms to vocabulary (if not present)
new_tokens = [term for term in medical_terms if term not in processor.tokenizer.get_vocab()]
processor.tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(processor.tokenizer))

# Configure LoRA
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "lm_head"],
    lora_dropout=0.1,
    bias="none",
)

model = get_peft_model(model, lora_config)

# Training (similar to Part 1)
# ... training code ...
```

### Step 3: Test Medical Terminology Accuracy

```python
import jiwer

def calculate_wer(reference, hypothesis):
    return jiwer.wer(reference, hypothesis)

# Test on medical terminology
test_cases = [
    ("Patient has pancolitis", "Patient has pancolitis"),
    ("Hematochezia present", "Hematochezia present"),
    ("Vedolizumab therapy", "Vedolizumab therapy"),
]

for ref, hyp in test_cases:
    wer = calculate_wer(ref, hyp)
    print(f"WER: {wer:.2%}")
```

---

## Part 3: Fine-Tuning MedLlama for HPI/Assessment Extraction

### Step 1: Prepare HPI/Assessment Dataset

```python
# Format: data/hpi_assessment_train.jsonl
{
  "instruction": "Extract HPI and Assessment from this medical conversation transcript.",
  "input": "Doctor: What brings you in today? Patient: I've been having abdominal pain for three days. Doctor: Any blood in stool? Patient: Yes, some bright red blood. Doctor: When did it start? Patient: About a week ago.",
  "output": "HPI: 50-year-old male presents with 3-day history of abdominal pain and 1-week history of bright red blood per rectum (hematochezia). Symptoms are ongoing.\n\nAssessment: 1. Lower GI bleeding, likely from inflammatory bowel disease or diverticular disease. 2. Abdominal pain, acute onset."
}
```

### Step 2: Fine-Tune MedLlama with QLoRA

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch
from bitsandbytes import BitsAndBytesConfig

# Load model with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model_name = "medllama2:7b-instruct"  # Or use HuggingFace path
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Prepare model for training
model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Load dataset
def format_prompt(example):
    return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""

dataset = load_dataset("json", data_files=f"{DATA_PATH}/hpi_assessment_train.jsonl", split="train")

def tokenize_function(examples):
    texts = [format_prompt(ex) for ex in examples]
    return tokenizer(texts, truncation=True, max_length=2048, padding="max_length")

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./medllama-hpi-assessment",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_steps=100,
    num_train_epochs=3,
    fp16=True,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# Train
trainer.train()

# Save
model.save_pretrained("./medllama-hpi-assessment-final")
```

### Step 3: Test HPI/Assessment Extraction

```python
def test_hpi_extraction(model, tokenizer, transcript):
    prompt = f"""### Instruction:
Extract HPI and Assessment from this medical conversation transcript.

### Input:
{transcript}

### Response:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract HPI and Assessment
    if "HPI:" in response:
        hpi = response.split("HPI:")[1].split("Assessment:")[0].strip()
    if "Assessment:" in response:
        assessment = response.split("Assessment:")[1].strip()
    
    return {"hpi": hpi, "assessment": assessment}

# Test
test_transcript = "Doctor: What brings you in? Patient: Abdominal pain for 3 days. Doctor: Any blood? Patient: Yes."
result = test_hpi_extraction(model, tokenizer, test_transcript)
print(f"HPI: {result['hpi']}")
print(f"Assessment: {result['assessment']}")
```

---

## Data Preparation

### Creating Speaker-Diarized Transcripts

```python
# Use pyannote.audio for automatic speaker diarization
!pip install -q pyannote.audio

from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

def diarize_audio(audio_path):
    diarization = pipeline(audio_path)
    
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })
    
    return segments

# Combine with Whisper transcription
def create_diarized_transcript(audio_path):
    # 1. Diarize speakers
    segments = diarize_audio(audio_path)
    
    # 2. Transcribe each segment
    transcript_parts = []
    for seg in segments:
        audio_segment = extract_audio_segment(audio_path, seg["start"], seg["end"])
        text = transcribe(audio_segment)
        transcript_parts.append(f"{seg['speaker']}: {text}")
    
    return " ".join(transcript_parts)
```

### Creating HPI/Assessment Training Pairs

```python
# Extract HPI and Assessment from structured summaries
def extract_hpi_assessment(summary_text):
    hpi = ""
    assessment = ""
    
    if "HPI:" in summary_text:
        hpi = summary_text.split("HPI:")[1].split("Assessment:")[0].strip()
    if "Assessment:" in summary_text:
        assessment = summary_text.split("Assessment:")[1].split("Plan:")[0].strip()
    
    return {"hpi": hpi, "assessment": assessment}

# Create training pairs
training_pairs = []
for transcript, summary in zip(transcripts, summaries):
    hpi_assessment = extract_hpi_assessment(summary)
    training_pairs.append({
        "input": transcript,
        "output": f"HPI: {hpi_assessment['hpi']}\n\nAssessment: {hpi_assessment['assessment']}"
    })
```

---

## Deployment

### Step 1: Download Fine-Tuned Models

```python
# Download from Colab to local machine
from google.colab import files

# Download Whisper model
!zip -r whisper-speaker-diarized-final.zip whisper-speaker-diarized-final
files.download('whisper-speaker-diarized-final.zip')

# Download MedLlama model
!zip -r medllama-hpi-assessment-final.zip medllama-hpi-assessment-final
files.download('medllama-hpi-assessment-final.zip')
```

### Step 2: Integrate into MedRec

1. **Update Whisper Model:**
   - Place fine-tuned model in `models/faster-whisper/`
   - Update `config.json`:
   ```json
   {
     "whisper": {
       "faster_model": "whisper-speaker-diarized",
       "engine": "faster"
     }
   }
   ```

2. **Update MedLlama Model:**
   - Load fine-tuned model in Ollama:
   ```bash
   ollama create medllama2-gi-hpi --file Modelfile
   ```
   - Update `config.json`:
   ```json
   {
     "summarizer": {
       "model": "medllama2-gi-hpi"
     }
   }
   ```

---

## Accuracy Improvements

### Expected Improvements After Fine-Tuning

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Medical Terminology WER** | 15-20% | 5-8% | 60-70% reduction |
| **Speaker Diarization Accuracy** | N/A | 85-90% | New capability |
| **HPI Extraction Accuracy** | 70-75% | 90-95% | 20-25% improvement |
| **Assessment Accuracy** | 75-80% | 92-97% | 15-20% improvement |

### Validation Metrics

```python
# Calculate accuracy metrics
def calculate_accuracy(predictions, references):
    # HPI extraction accuracy
    hpi_accuracy = calculate_hpi_f1(predictions['hpi'], references['hpi'])
    
    # Assessment accuracy
    assessment_accuracy = calculate_assessment_f1(predictions['assessment'], references['assessment'])
    
    # Speaker diarization accuracy
    speaker_accuracy = calculate_speaker_accuracy(predictions['speakers'], references['speakers'])
    
    return {
        'hpi_f1': hpi_accuracy,
        'assessment_f1': assessment_accuracy,
        'speaker_accuracy': speaker_accuracy
    }
```

---

## Next Steps

1. **Collect Real Data:** Record 20-50 hours of actual medical dictations
2. **Iterate:** Fine-tune on real data, measure improvements
3. **Validate:** Test with doctors, collect feedback
4. **Deploy:** Integrate fine-tuned models into production

---

## Resources

- [Whisper Fine-Tuning Guide](https://huggingface.co/docs/transformers/model_doc/whisper)
- [MedLlama Model Card](https://huggingface.co/medllama2)
- [LoRA Fine-Tuning](https://huggingface.co/docs/peft)
- [Google Colab Pro](https://colab.research.google.com/signup) (for longer training)

---

**Last Updated:** December 17, 2025


