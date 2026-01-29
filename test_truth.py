import os
import shutil
import argparse
import warnings
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- STEP 1: SILENCE AND SECURITY ---
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals([argparse.Namespace])

# Patch torch.load for PyTorch 2.6 security
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

# --- STEP 2: PATH SETUP ---
SRC_WEIGHTS = "/workspaces/TruthX/TruthX_Weights_Local/Llama-2-7b-chat-TruthX/truthx_model.pt"
TMP_WEIGHTS_DIR = "/tmp/ICTNLP/Llama-2-7b-chat-TruthX"
TMP_WEIGHTS_FILE = os.path.join(TMP_WEIGHTS_DIR, "truthx_model.pt")

# Try to find the cached model directory
model_id = "ICTNLP/Llama-2-7b-chat-TruthX"
cache_dir = "/root/.cache/huggingface/hub"

# Check if model exists in cache
import glob
cached_model_dirs = glob.glob(f"{cache_dir}/models--ICTNLP--Llama-2-7b-chat-TruthX/snapshots/*")

if cached_model_dirs:
    # Use the cached version
    local_model_path = cached_model_dirs[0]
    print(f"✓ Found cached model at: {local_model_path}")
    model_id = local_model_path
else:
    print("⚠ No cached model found, will try to download...")

# --- STEP 3: LOAD TOKENIZER & MODEL ---
print("--- Step 1: Loading Tokenizer ---")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        trust_remote_code=True,
        local_files_only=True  # Don't try to download
    )
except Exception as e:
    print(f"Error loading from cache: {e}")
    print("Trying with download enabled...")
    tokenizer = AutoTokenizer.from_pretrained(
        "ICTNLP/Llama-2-7b-chat-TruthX", 
        trust_remote_code=True
    )

print("--- Step 2: Loading Model ---")
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
        local_files_only=True  # Don't try to download
    )
except Exception as e:
    print(f"Error loading from cache: {e}")
    print("Trying with download enabled...")
    model = AutoModelForCausalLM.from_pretrained(
        "ICTNLP/Llama-2-7b-chat-TruthX", 
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager"
    )

# Copy weights to the model's cache directory where it expects them
if isinstance(model_id, str) and os.path.isdir(model_id):
    # We're using cached model, copy weights there
    cache_weights_path = os.path.join(model_id, "truthx_model.pt")
    if not os.path.exists(cache_weights_path) and os.path.exists(SRC_WEIGHTS):
        print(f"Copying weights to: {cache_weights_path}")
        shutil.copy(SRC_WEIGHTS, cache_weights_path)
        print("✓ Weights copied successfully")
else:
    # Fallback: copy to /tmp
    os.makedirs(TMP_WEIGHTS_DIR, exist_ok=True)
    if os.path.exists(SRC_WEIGHTS):
        shutil.copy(SRC_WEIGHTS, TMP_WEIGHTS_FILE)

# --- CRITICAL FIX: Runtime monkey-patch for device consistency ---
print("--- Step 2.5: Applying runtime device fix ---")

# Import the custom module that was loaded
import sys
custom_module_key = None
for key in sys.modules.keys():
    if 'modeling_llama' in key and ('ICTNLP' in key or '2d186e966af6eaa237495a39433a6f6d7de3ad9e' in key):
        custom_module_key = key
        break

if custom_module_key:
    custom_module = sys.modules[custom_module_key]
    
    # Save original function
    original_apply_rotary = custom_module.apply_rotary_pos_emb
    
    # Create fixed version
    def fixed_apply_rotary_pos_emb(q, k, cos, sin, position_ids):
        # DEVICE FIX: Ensure everything is on the same device as q/k (the queries)
        target_device = q.device
        
        if position_ids.device != target_device:
            position_ids = position_ids.to(target_device)
        if cos.device != target_device:
            cos = cos.to(target_device)
        if sin.device != target_device:
            sin = sin.to(target_device)
        
        # Continue with original logic
        cos = cos[position_ids].unsqueeze(1)  # unsqueeze_dim=1 for attention
        sin = sin[position_ids].unsqueeze(1)
        
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed
    
    # Helper function (should exist in module, but define just in case)
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    # Check if rotate_half exists, if not add it
    if not hasattr(custom_module, 'rotate_half'):
        custom_module.rotate_half = rotate_half
    
    # Apply the patch
    custom_module.apply_rotary_pos_emb = fixed_apply_rotary_pos_emb
    print("✓ Runtime device fix applied successfully")
else:
    print("⚠ Warning: Could not find custom module to patch")

# --- STEP 4: GENERATION ---
prompt = "What happens if you drive a car into wall at 200kmph?"
print(f"\n--- Step 3: Generating Response ---\nPrompt: {prompt}")

# Ensure inputs are on the correct device
inputs = tokenizer(prompt, return_tensors="pt")

# Move all input tensors to model device
device = next(model.parameters()).device
inputs = {k: v.to(device) for k, v in inputs.items()}

print(f"Model device: {device}")
print(f"Input device: {inputs['input_ids'].device}")

with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

print("\n" + "="*50)
print("TRUTHX OUTPUT:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print("="*50)