import torch

# Patch for Windows/Unsloth/torchao compatibility
for i in range(1, 8):
    if not hasattr(torch, f"int{i}"):
        setattr(torch, f"int{i}", torch.int8)

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

def main():
    # 1. Configuration
    model_name = "unsloth/llama-3-8b-bnb-4bit"
    max_seq_length = 2048
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage.

    # 2. Load Model & Tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # 3. Add LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Rank, higher = more parameters
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Optimized to 0
        bias = "none",    # Optimized to "none"
        use_gradient_checkpointing = "unsloth", # 4x longer contexts, 0 memory
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    # 4. Data Preparation
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Llama 3 Prompt Format
            text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>"
            texts.append(text)
        return { "text" : texts, }

    dataset = load_dataset("json", data_files="data/gi_seed_dataset.jsonl", split="train")
    dataset = dataset.map(formatting_prompts_func, batched = True,)

    # 5. Trainer Configuration
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 0, # Setting to 0 on Windows to avoid complications
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 8,
            warmup_steps = 5,
            max_steps = 60, # Small dataset, small number of steps
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "models/gio_lora",
        ),
    )

    # 6. Training
    print("Starting fine-tuning...")
    trainer_stats = trainer.train()

    # 7. Saving and Merging
    print("Saving LoRA adapters...")
    model.save_pretrained_lora("models/gio_lora_final") 
    print("Fine-tuning complete. Model saved to models/gio_lora_final")

if __name__ == "__main__":
    main()
