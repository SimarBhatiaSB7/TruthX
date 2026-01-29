#!/usr/bin/env python3
"""
TruthX Interactive Inference
Ask unlimited questions without reloading the model
"""

import os
import sys
import argparse
import warnings
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import glob
import shutil

# --- APPLY ALL PATCHES FIRST ---
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

print("\n🔄 Loading TruthX model (please wait 1-2 minutes)...", end='', flush=True)

SRC_WEIGHTS = "/workspaces/TruthX/TruthX_Weights_Local/Llama-2-7b-chat-TruthX/truthx_model.pt"
model_id = "ICTNLP/Llama-2-7b-chat-TruthX"
cache_dir = "/root/.cache/huggingface/hub"

# Find cached model
cached_model_dirs = glob.glob(f"{cache_dir}/models--ICTNLP--Llama-2-7b-chat-TruthX/snapshots/*")
if cached_model_dirs:
    local_model_path = cached_model_dirs[0]
    model_id = local_model_path

# Load tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
except:
    tokenizer = AutoTokenizer.from_pretrained("ICTNLP/Llama-2-7b-chat-TruthX", trust_remote_code=True)

# Load model
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        local_files_only=True
    )
except:
    model = AutoModelForCausalLM.from_pretrained(
        "ICTNLP/Llama-2-7b-chat-TruthX",
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager"
    )

# Setup weights
if isinstance(model_id, str) and os.path.isdir(model_id):
    cache_weights_path = os.path.join(model_id, "truthx_model.pt")
    if not os.path.exists(cache_weights_path) and os.path.exists(SRC_WEIGHTS):
        shutil.copy(SRC_WEIGHTS, cache_weights_path)

# Apply runtime fixes
custom_module_key = None
for key in sys.modules.keys():
    if 'modeling_llama' in key and ('ICTNLP' in key or '2d186e966af6eaa237495a39433a6f6d7de3ad9e' in key):
        custom_module_key = key
        break

if custom_module_key:
    custom_module = sys.modules[custom_module_key]
    
    def fixed_apply_rotary_pos_emb(q, k, cos, sin, position_ids):
        target_device = q.device
        if position_ids.device != target_device:
            position_ids = position_ids.to(target_device)
        if cos.device != target_device:
            cos = cos.to(target_device)
        if sin.device != target_device:
            sin = sin.to(target_device)
        
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
        
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)
        
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed
    
    custom_module.apply_rotary_pos_emb = fixed_apply_rotary_pos_emb

device = next(model.parameters()).device
print(f" ✓\n✅ TruthX model ready on {device}!")

# --- INTERACTIVE LOOP ---
print("\n" + "=" * 70)
print("Commands: 'quit' to exit | 'clear' to clear screen | 'settings' for options")
print("=" * 70)

# Default settings
max_tokens = 256
temperature = 0.7

while True:
    try:
        user_input = input("\n💬 Ask: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if user_input.lower() == 'clear':
            os.system('clear' if os.name != 'nt' else 'cls')
            print("=" * 70)
            print("Commands: 'quit' to exit | 'clear' to clear screen | 'settings' for options")
            print("=" * 70)
            continue
        
        if user_input.lower() == 'settings':
            print(f"\n⚙️  Current: max_tokens={max_tokens}, temperature={temperature}")
            try:
                new_max = input(f"   Max tokens [{max_tokens}]: ").strip()
                if new_max:
                    max_tokens = int(new_max)
                new_temp = input(f"   Temperature [{temperature}]: ").strip()
                if new_temp:
                    temperature = float(new_temp)
                print(f"   ✓ Updated")
            except ValueError:
                print("   ⚠️  Invalid input, keeping current settings")
            continue
        
        # Generate response
        inputs = tokenizer(user_input, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the answer
        if user_input in response:
            response = response.replace(user_input, "").strip()
        
        # Clean output
        print(f"\n✨ {response}\n")
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        break
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Continuing...\n")

# Cleanup
print("\n🧹 Cleaning up...")
del model
del tokenizer
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("✓ Done")