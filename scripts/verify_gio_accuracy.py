
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel, PeftConfig

def main():
    # 1. Configuration
    base_model_name = "unsloth/llama-3-8b-bnb-4bit"
    adapter_model_name = "models/gio_lora_standard"
    
    print(f"Loading base model: {base_model_name}...")
    
    # 2. Load Base Model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 3. Load Adapter
    print(f"Loading adapter: {adapter_model_name}...")
    try:
        model = PeftModel.from_pretrained(base_model, adapter_model_name)
    except Exception as e:
        print(f"Error loading adapter: {e}")
        print("Running with base model only for comparison...")
        model = base_model

    # 4. Test Inference
    test_cases = [
        {
            "input": "Doctor: Hi, tell me about the tummy pain.\nPatient: It's mostly after I eat spicy food, right here in the middle.\nDoctor: Does it burn?\nPatient: Yeah, like a fire. And I get this sour taste in my throat.\nDoctor: sounds like GERD. I'll prescribe Omeprazole.",
            "instruction": "Convert the following GI consultation transcript into a professional clinical note."
        },
        {
           "input": "Doctor: How are the bloody stools?\nPatient: They stopped after the steroids.\nDoctor: Good. Any fever or joint pain?\nPatient: My knees hurt a bit, but no fever.\nDoctor: Okay, we'll taper the prednisone and start Humira.",
           "instruction": "Convert the following GI consultation transcript into a professional clinical note."
        }
    ]
    
    print("\n--- Starting Inference Verification ---\n")
    
    for i, test in enumerate(test_cases):
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{test['instruction']}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{test['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256, 
                use_cache=True, 
                temperature=0.1,  # Low temperature for factual notes
                do_sample=True,
                top_p=0.9
            )
            
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the assistant response if possible, or print whole thing
        response = generated_text.split("assistant\n\n")[-1] if "assistant\n\n" in generated_text else generated_text
        
        print(f"Case {i+1} Input:\n{test['input'][:100]}...")
        print(f"Case {i+1} Output:\n{response}\n")
        print("-" * 50)

if __name__ == "__main__":
    main()
