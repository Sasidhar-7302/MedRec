# Google Colab Fine-Tuning Guide for MedRec

## Overview
This guide walks you through fine-tuning Whisper and the summarization model using Google Colab's free GPU resources.

---

## Part 1: Whisper Fine-Tuning

### Why Fine-Tune Whisper?
- Improve medical terminology recognition (e.g., "pancolitis", "vedolizumab")
- Better accuracy for GI-specific terms
- Reduced transcription errors

### Prerequisites
1. Google account
2. Medical dictation audio files (WAV format, 16kHz, mono)
3. Corresponding transcripts (text files)

---

## Step 1: Prepare Your Data

### 1.1 Collect Audio-Transcript Pairs
You need at least 20-50 high-quality pairs:
- Each audio file: 30 seconds to 5 minutes
- Clear audio quality
- Corresponding accurate transcripts

### 1.2 Organize Data Structure
```
training_data/
├── audio/
│   ├── dictation_001.wav
│   ├── dictation_002.wav
│   └── ...
└── transcripts/
    ├── dictation_001.txt
    ├── dictation_002.txt
    └── ...
```

### 1.3 Create Training Manifest
Create a JSONL file (one JSON object per line):
```json
{"audio_filepath": "audio/dictation_001.wav", "text": "Patient presents with abdominal pain..."}
{"audio_filepath": "audio/dictation_002.wav", "text": "Follow-up for Crohn's disease..."}
```

---

## Step 2: Google Colab Setup

### 2.1 Create New Colab Notebook
1. Go to https://colab.research.google.com
2. File → New Notebook
3. Name it: "MedRec_Whisper_FineTuning"

### 2.2 Install Dependencies
```python
# First cell - Install required packages
!pip install -q openai-whisper
!pip install -q datasets
!pip install -q transformers
!pip install -q accelerate
!pip install -q librosa
!pip install -q jiwer  # For evaluation
```

### 2.3 Upload Your Data
```python
# Second cell - Upload training data
from google.colab import files
import zipfile
import os

# Option 1: Upload zip file
uploaded = files.upload()
for fn in uploaded.keys():
    if fn.endswith('.zip'):
        with zipfile.ZipFile(fn, 'r') as zip_ref:
            zip_ref.extractall('training_data')
        print(f'Extracted {fn}')

# Option 2: Upload individual files
# Use Colab's file browser (left sidebar) to upload files
```

---

## Step 3: Fine-Tune Whisper Model

### 3.1 Load Base Model
```python
# Third cell - Load Whisper model
import whisper
import torch

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load base model (start with small or medium)
model = whisper.load_model("small", device=device)
print("Model loaded successfully")
```

### 3.2 Prepare Dataset
```python
# Fourth cell - Prepare training data
import json
import os
from pathlib import Path

def load_training_data(manifest_path):
    """Load training data from JSONL manifest."""
    data = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# Load your manifest
training_data = load_training_data('training_data/manifest.jsonl')
print(f"Loaded {len(training_data)} training samples")
```

### 3.3 Fine-Tuning Script
```python
# Fifth cell - Fine-tuning
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class WhisperDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        audio_path = item['audio_filepath']
        text = item['text']
        
        # Load and preprocess audio
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        
        # Tokenize text
        tokens = self.tokenizer.encode(text)
        
        return mel, torch.tensor(tokens)

# Initialize dataset
tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual)
dataset = WhisperDataset(training_data, tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch_idx, (mels, tokens) in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        
        # Forward pass
        # Note: This is simplified - actual Whisper fine-tuning is more complex
        # You may want to use HuggingFace's Whisper fine-tuning scripts
        
        # For production, use: https://github.com/huggingface/transformers/tree/main/examples/pytorch/speech-recognition
        
        loss = criterion(output, tokens)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")
```

### 3.4 Alternative: Use HuggingFace Transformers (Recommended)
```python
# Better approach - Use HuggingFace's fine-tuning script
!git clone https://github.com/huggingface/transformers.git
%cd transformers
!pip install -q -e .

# Use the official fine-tuning script
!python examples/pytorch/speech-recognition/run_speech_recognition_seq2seq.py \
    --model_name_or_path="openai/whisper-small" \
    --dataset_name="your_dataset" \
    --output_dir="./whisper-medical-finetuned" \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps=2 \
    --learning_rate=1e-5 \
    --warmup_steps=500 \
    --max_steps=5000 \
    --gradient_checkpointing \
    --fp16 \
    --evaluation_strategy="steps" \
    --per_device_eval_batch_size=8 \
    --predict_with_generate \
    --generation_max_length=225 \
    --save_steps=1000 \
    --eval_steps=1000 \
    --logging_steps=25 \
    --report_to="tensorboard" \
    --load_best_model_at_end \
    --metric_for_best_model="wer" \
    --greater_is_better=False \
    --push_to_hub
```

---

## Part 2: Summarization Model Fine-Tuning

### Why Fine-Tune Summarization?
- Better structured output (Findings/Assessment/Plan)
- Improved medical terminology usage
- More consistent formatting

### Step 1: Prepare Training Data
Create pairs of transcripts and ideal summaries:

```
training_pairs/
├── transcript_001.txt
├── summary_001.txt
├── transcript_002.txt
├── summary_002.txt
└── ...
```

### Step 2: Fine-Tune with LoRA (Parameter-Efficient)
```python
# Install PEFT for LoRA
!pip install -q peft bitsandbytes accelerate

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import json

# Load base model (MedLlama or Llama 2)
model_name = "meta-llama/Llama-2-7b-chat-hf"  # Or your MedLlama model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,  # Use 8-bit quantization
    device_map="auto"
)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,  # Rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]  # Target attention layers
)

model = get_peft_model(model, lora_config)

# Prepare dataset
def prepare_dataset(transcript_path, summary_path):
    with open(transcript_path, 'r') as f:
        transcript = f.read()
    with open(summary_path, 'r') as f:
        summary = f.read()
    
    prompt = f"""You are a clinical summarization assistant. Transform this transcript into a structured clinical note.

Transcript:
{transcript}

Summary:
{summary}"""
    
    return tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

# Training
training_args = TrainingArguments(
    output_dir="./medrec-summarizer-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

---

## Part 3: Export and Deploy

### 3.1 Export Fine-Tuned Whisper Model
```python
# Save model
model.save_pretrained("./whisper-medical-finetuned")
tokenizer.save_pretrained("./whisper-medical-finetuned")

# Convert to GGML format for whisper.cpp (if needed)
# Use whisper.cpp's convert-pt-to-ggml.py script
```

### 3.2 Export Summarization Model
```python
# Merge LoRA weights
model = model.merge_and_unload()

# Save merged model
model.save_pretrained("./medrec-summarizer-final")
tokenizer.save_pretrained("./medrec-summarizer-final")

# Quantize for Ollama (if using GGUF)
# Use llama.cpp quantization tools
```

### 3.3 Download Models
```python
# Download to your local machine
from google.colab import files

# Create zip
!zip -r medrec_models.zip whisper-medical-finetuned medrec-summarizer-final

# Download
files.download('medrec_models.zip')
```

---

## Part 4: Integration with MedRec

### 4.1 Update Config for Fine-Tuned Whisper
Edit `config.json`:
```json
{
  "whisper": {
    "model_path": "models/whisper/ggml-small-medical.bin",
    "faster_model": "your-finetuned-model",
    ...
  }
}
```

### 4.2 Update Summarizer Model
```json
{
  "summarizer": {
    "model": "medrec-summarizer-finetuned:7b",
    ...
  }
}
```

---

## Tips for Best Results

### Data Collection
1. **Quality over Quantity**: 20-30 high-quality pairs > 100 poor pairs
2. **Diverse Cases**: Include various GI conditions, procedures, follow-ups
3. **Accurate Transcripts**: Manually verify all transcripts
4. **Audio Quality**: Use good microphones, minimize background noise

### Training
1. **Start Small**: Begin with 3-5 epochs, evaluate, then adjust
2. **Monitor Loss**: Watch for overfitting (validation loss increases)
3. **Learning Rate**: Start with 1e-5, adjust based on convergence
4. **Batch Size**: Use largest batch that fits in GPU memory

### Evaluation
1. **Word Error Rate (WER)**: Target < 10% for transcription
2. **ROUGE Scores**: For summarization quality
3. **Manual Review**: Always manually check sample outputs

---

## Resources

- **Whisper Fine-Tuning**: https://github.com/openai/whisper
- **HuggingFace Whisper**: https://huggingface.co/docs/transformers/model_doc/whisper
- **LoRA Fine-Tuning**: https://github.com/huggingface/peft
- **Ollama Models**: https://ollama.com/library

---

## Troubleshooting

### Out of Memory
- Reduce batch size
- Use gradient checkpointing
- Use 8-bit quantization
- Use smaller base model

### Poor Results
- Check data quality
- Increase training data
- Adjust learning rate
- Try different base models

### Slow Training
- Use GPU (Colab Pro recommended)
- Reduce batch size
- Use mixed precision (fp16)

---

*Last Updated: 2025-01-18*



