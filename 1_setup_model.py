import os
import shutil

print("="*60)
print("TruthX Model Setup - Run this ONCE")
print("="*60)

# --- STEP 1: PATH SETUP ---
SRC_WEIGHTS = "/workspaces/TruthX/TruthX_Weights_Local/Llama-2-7b-chat-TruthX/truthx_model.pt"
CACHE_WEIGHTS_DIR = "/root/.cache/huggingface/hub/models--ICTNLP--Llama-2-7b-chat-TruthX/snapshots"
LOCAL_WEIGHTS_DIR = "/workspaces/TruthX/ICTNLP/Llama-2-7b-chat-TruthX"

# Create local weights directory in working directory
print("\n[1/3] Setting up weights directory...")
os.makedirs(LOCAL_WEIGHTS_DIR, exist_ok=True)
local_weights_file = os.path.join(LOCAL_WEIGHTS_DIR, "truthx_model.pt")

if os.path.exists(SRC_WEIGHTS):
    if not os.path.exists(local_weights_file):
        shutil.copy(SRC_WEIGHTS, local_weights_file)
    print(f"✓ Weights copied to {local_weights_file}")
else:
    print(f"⚠ Warning: Source weights not found at {SRC_WEIGHTS}")

# --- STEP 2: PATCH THE CUSTOM MODEL FILE ---
print("\n[2/3] Patching custom model for device consistency...")

custom_module_path = "/root/.cache/huggingface/modules/transformers_modules/ICTNLP/Llama-2-7b-chat-TruthX/2d186e966af6eaa237495a39433a6f6d7de3ad9e/modeling_llama.py"

if not os.path.exists(custom_module_path):
    print(f"⚠ Custom model file not found at: {custom_module_path}")
    print("   Run the inference script once to download it, then run this setup again.")
    exit(1)

# Read the file
with open(custom_module_path, 'r') as f:
    content = f.read()

# Check if already patched
if 'DEVICE_FIX_APPLIED' in content:
    print("✓ Model already patched!")
else:
    print("  Applying device compatibility patch...")
    
    # Find the line that causes the error
    old_line = '    cos = cos[position_ids].unsqueeze(unsqueeze_dim)'
    
    # Check if the line exists
    if old_line in content:
        # Create the patch
        patch = '''    # DEVICE_FIX_APPLIED - Ensure all tensors on same device
    target_device = q.device
    if position_ids.device != target_device:
        position_ids = position_ids.to(target_device)
    if cos.device != target_device:
        cos = cos.to(target_device)
    if sin.device != target_device:
        sin = sin.to(target_device)
    
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)'''
        
        # Replace
        content = content.replace(old_line, patch)
        
        # Write back
        with open(custom_module_path, 'w') as f:
            f.write(content)
        
        print("✓ Patch applied successfully!")
    else:
        print("⚠ Could not find target line to patch.")
        print("  Trying alternative patching method...")
        
        # Alternative: Add device fix at the start of apply_rotary_pos_emb
        if 'def apply_rotary_pos_emb(q, k, cos, sin, position_ids):' in content:
            old_func_start = 'def apply_rotary_pos_emb(q, k, cos, sin, position_ids):'
            new_func_start = '''def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # DEVICE_FIX_APPLIED
    target_device = q.device
    if position_ids.device != target_device:
        position_ids = position_ids.to(target_device)
    if cos.device != target_device:
        cos = cos.to(target_device)
    if sin.device != target_device:
        sin = sin.to(target_device)'''
            
            content = content.replace(old_func_start, new_func_start, 1)
            
            with open(custom_module_path, 'w') as f:
                f.write(content)
            
            print("✓ Alternative patch applied!")
        else:
            print("❌ Could not patch automatically.")
            print("   Manual patching required - see instructions below.")

print("\n" + "="*60)
print("Setup complete!")
print("\nIMPORTANT: Run inference from /workspaces/TruthX directory")
print("Command: cd /workspaces/TruthX && python 2_run_inference.py")
print("="*60)