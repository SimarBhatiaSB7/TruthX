import os
import sys
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

# Patch torch.load
original_load = torch.load
def patched_load(f, *args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(f, *args, **kwargs)
torch.load = patched_load

# Patch .to() for 4-bit
original_to = transformers.modeling_utils.PreTrainedModel.to
def patched_to(self, *args, **kwargs):
    if getattr(self, "is_quantized", False) or getattr(self, "quantization_method", None):
        return self
    return original_to(self, *args, **kwargs)
transformers.modeling_utils.PreTrainedModel.to = patched_to

# --- ARGUMENT PARSER ---
parser = argparse.ArgumentParser(description='TruthX Single Query')
parser.add_argument('prompt', type=str, help='Your question or prompt')
parser.add_argument('--max_tokens', type=int, default=100, help='Max tokens to generate')
parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
args = parser.parse_args()

# --- CONFIG ---
model_id = "ICTNLP/Llama-2-7b-chat-TruthX"

print("Loading model (this takes ~4 minutes)...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

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

# --- GENERATE ---
print(f"\nPrompt: {args.prompt}\n")
print("Generating response...")

inputs = tokenizer(args.prompt, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=args.max_tokens,
        do_sample=True,
        temperature=args.temperature,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n" + "="*60)
print("RESPONSE:")
print("="*60)
print(response)
print("="*60)