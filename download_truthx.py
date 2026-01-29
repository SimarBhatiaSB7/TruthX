#!/usr/bin/env python3
"""
TruthX Complete Setup - All-in-One Script
Fixes transformers, downloads model, and verifies everything works
"""

import subprocess
import sys
import os

print("=" * 70)
print("          TruthX Complete Setup")
print("=" * 70)
print("\nThis script will:")
print("  1. Fix transformers installation")
print("  2. Download TruthX model (~13GB)")
print("  3. Verify everything works")
print("\nThis may take 10-30 minutes depending on your connection.")
print("=" * 70)

response = input("\nContinue? (yes/no): ").strip().lower()
if response not in ['yes', 'y']:
    print("Setup cancelled.")
    sys.exit(0)

def run_command(cmd, description, ignore_errors=False):
    """Run a shell command and show output"""
    print(f"\n{description}")
    print("-" * 70)
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=False
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        if result.returncode != 0 and not ignore_errors:
            print(f"⚠️  Command failed with exit code {result.returncode}")
            return False
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# STEP 1: Fix transformers
print("\n" + "=" * 70)
print("STEP 1: Fixing transformers installation")
print("=" * 70)

run_command("pip uninstall -y transformers", "Removing old transformers...")

success = run_command(
    "pip install transformers==4.40.0 --break-system-packages",
    "Installing transformers 4.40.0...",
    ignore_errors=True
)

if not success:
    run_command(
        "pip install transformers==4.40.0",
        "Retrying without --break-system-packages..."
    )

run_command(
    "pip install accelerate --break-system-packages --upgrade || pip install accelerate --upgrade",
    "Installing/updating accelerate...",
    ignore_errors=True
)

# Verify transformers
sys.path = [p for p in sys.path if 'transformers_old' not in p]

try:
    if 'transformers' in sys.modules:
        del sys.modules['transformers']
    
    import transformers
    print(f"\n✓ Transformers {transformers.__version__} installed")
    
    from transformers.cache_utils import Cache
    print("✓ cache_utils available")
except Exception as e:
    print(f"\n❌ Transformers verification failed: {e}")
    sys.exit(1)

# STEP 2: Download model
print("\n" + "=" * 70)
print("STEP 2: Downloading TruthX model")
print("=" * 70)

MODEL_ID = "ICTNLP/Llama-2-7b-chat-TruthX"
LOCAL_DIR = "./TruthX-model"

if os.path.exists(LOCAL_DIR) and os.path.exists(os.path.join(LOCAL_DIR, "config.json")):
    print(f"✓ Model already exists at {LOCAL_DIR}")
    print("  Skipping download...")
else:
    try:
        from huggingface_hub import snapshot_download
        
        print(f"\nDownloading {MODEL_ID}...")
        print("This will download ~13GB - please be patient!")
        
        model_path = snapshot_download(
            repo_id=MODEL_ID,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"\n✓ Model downloaded to: {model_path}")
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nYou can download manually with:")
        print(f"  huggingface-cli download {MODEL_ID} --local-dir {LOCAL_DIR}")
        sys.exit(1)

# STEP 3: Verify model loads
print("\n" + "=" * 70)
print("STEP 3: Verifying model loads correctly")
print("=" * 70)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_DIR,
        trust_remote_code=True,
        local_files_only=True
    )
    print("✓ Tokenizer loaded")
    
    print("\nLoading model (CPU only for verification)...")
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_DIR,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cpu",
        local_files_only=True
    )
    print("✓ Model loaded successfully")
    
    # Quick tokenization test
    test_input = "Hello, how are you?"
    inputs = tokenizer(test_input, return_tensors="pt")
    print(f"✓ Tokenization works ({inputs['input_ids'].shape[1]} tokens)")
    
    # Cleanup
    del model
    del tokenizer
    import gc
    gc.collect()
    
except Exception as e:
    print(f"\n❌ Model verification failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# SUCCESS!
print("\n" + "=" * 70)
print("✓ SETUP COMPLETE!")
print("=" * 70)
print(f"\nModel location: {os.path.abspath(LOCAL_DIR)}")
print("\nYou can now run TruthX inference with:")
print("  python run_truthx_local.py")
print("\nThe model will load once and stay in memory for fast queries.")
print("=" * 70)