import os
import argparse
import warnings
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- SILENCE WARNINGS ---
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals([argparse.Namespace])

# Patch torch.load for PyTorch 2.6
original_load = torch.load
def patched_load(f, *args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(f, *args, **kwargs)
torch.load = patched_load

# Patch .to() for 4-bit compatibility
original_to = transformers.modeling_utils.PreTrainedModel.to
def patched_to(self, *args, **kwargs):
    if getattr(self, "is_quantized", False) or getattr(self, "quantization_method", None):
        return self
    return original_to(self, *args, **kwargs)
transformers.modeling_utils.PreTrainedModel.to = patched_to

# --- CONFIG ---
model_id = "ICTNLP/Llama-2-7b-chat-TruthX"

print("="*60)
print("TruthX Inference - Interactive Mode")
print("="*60)

# Load model once
print("\n[1/3] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

print("\n[2/3] Loading model (this takes ~4 minutes)...")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    attn_implementation="eager"
)

device = next(model.parameters()).device
print(f"✓ Model loaded on: {device}")

print("\n[3/3] Ready for inference!")
print("="*60)

# --- INFERENCE LOOP ---
def generate_response(prompt, max_tokens=100, temperature=0.7):
    """Generate response for a given prompt"""
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Interactive mode
print("\nEnter your prompts (type 'quit' to exit):")
print("-"*60)

while True:
    try:
        prompt = input("\n🔮 Prompt: ")
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not prompt.strip():
            continue
        
        print("\n⏳ Generating response...")
        response = generate_response(prompt)
        
        print("\n" + "="*60)
        print("📝 RESPONSE:")
        print("-"*60)
        print(response)
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
        break
    except Exception as e:
        print(f"\n❌ Error: {e}")
        continue