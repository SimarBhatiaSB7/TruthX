#!/usr/bin/env python3
"""
TruthX Response Debugging Script
Helps identify why responses are incorrect
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Suppress warnings
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
warnings.filterwarnings("ignore")

# Patch torch.load for PyTorch 2.6
_original_torch_load = torch.load
def patched_torch_load(f, *args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(f, *args, **kwargs)
torch.load = patched_torch_load

MODEL_PATH = "/workspaces/TruthX/TruthX-model"

print("🔍 Loading model for debugging...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    local_files_only=True,
)

device = next(model.parameters()).device
print(f"✅ Model loaded on {device}\n")

# Test question
test_question = "What is 2+2?"

print(f"📝 Question: {test_question}")
print("=" * 70)

# Tokenize
inputs = tokenizer(test_question, return_tensors="pt").to(device)

print(f"\n🔍 Input IDs shape: {inputs['input_ids'].shape}")
print(f"🔍 Input tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")
print(f"🔍 Decoded input: '{tokenizer.decode(inputs['input_ids'][0])}'")

# Generate
print("\n🤖 Generating response...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

print(f"\n🔍 Output IDs shape: {outputs.shape}")
print(f"🔍 Output tokens: {tokenizer.convert_ids_to_tokens(outputs[0])}")

# Decode FULL output
full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\n📄 FULL decoded output:\n{full_response}")
print("=" * 70)

# Decode with method 1: Remove input from beginning
method1 = full_response.replace(test_question, "", 1).strip()
print(f"\n✂️  Method 1 (remove input once): '{method1}'")

# Decode with method 2: Only new tokens
new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
method2 = tokenizer.decode(new_tokens, skip_special_tokens=True)
print(f"\n✂️  Method 2 (only new tokens): '{method2}'")

# Decode with method 3: Split by input
if test_question in full_response:
    parts = full_response.split(test_question, 1)
    method3 = parts[1].strip() if len(parts) > 1 else full_response
    print(f"\n✂️  Method 3 (split by input): '{method3}'")
else:
    print(f"\n✂️  Method 3: Input not found in output!")

print("\n" + "=" * 70)
print("🎯 DIAGNOSIS:")
print("   Check which method gives the correct answer '4'")
print("   If none work, the model itself may have issues")
print("=" * 70)
