#!/usr/bin/env python3
"""
Comprehensive TruthX PyTorch 2.6 Patch
Fixes ALL torch.load() calls in the cached model files
"""

import os
import sys
import re

print("=" * 70)
print("     TruthX PyTorch 2.6 Comprehensive Patch")
print("=" * 70)

cache_dir = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/TruthX-model")

if not os.path.exists(cache_dir):
    print(f"\n❌ Cache directory not found: {cache_dir}")
    sys.exit(1)

print(f"\nCache directory: {cache_dir}")

# Files to patch
files_to_patch = ['truthx.py', 'modeling_llama.py']
patched_count = 0

for filename in files_to_patch:
    filepath = os.path.join(cache_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"\n⚠️  {filename} not found, skipping...")
        continue
    
    print(f"\n[Patching {filename}]")
    print("-" * 70)
    
    # Read file
    with open(filepath, 'r') as f:
        original_content = f.read()
    
    content = original_content
    
    # Method 1: Direct replacement of common patterns
    replacements = [
        ('torch.load(model_path)', 'torch.load(model_path, weights_only=False)'),
        ('torch.load(checkpoint_path)', 'torch.load(checkpoint_path, weights_only=False)'),
        ('torch.load(path)', 'torch.load(path, weights_only=False)'),
        ('torch.load(file)', 'torch.load(file, weights_only=False)'),
        ('torch.load(f)', 'torch.load(f, weights_only=False)'),
    ]
    
    for old, new in replacements:
        if old in content and 'weights_only' not in content[content.find(old):content.find(old)+100]:
            content = content.replace(old, new)
    
    # Method 2: Regex to catch any remaining torch.load without weights_only
    # Match torch.load(...) where ... doesn't contain weights_only
    def patch_torch_load(match):
        full_match = match.group(0)
        # Check if weights_only is already in this call
        if 'weights_only' in full_match:
            return full_match
        # Otherwise add it
        args = match.group(1)
        # If args already has keyword arguments, append
        if '=' in args:
            return f'torch.load({args}, weights_only=False)'
        # If it's just positional args, append
        else:
            return f'torch.load({args}, weights_only=False)'
    
    # Pattern to match torch.load with any arguments
    pattern = r'torch\.load\(([^)]+)\)'
    content = re.sub(pattern, patch_torch_load, content)
    
    # Count changes
    if content != original_content:
        # Write back
        with open(filepath, 'w') as f:
            f.write(content)
        
        # Count torch.load calls
        torch_loads = len(re.findall(r'torch\.load\(', content))
        weights_only = len(re.findall(r'weights_only=False', content))
        
        print(f"✓ Patched {filename}")
        print(f"  Total torch.load calls: {torch_loads}")
        print(f"  With weights_only=False: {weights_only}")
        patched_count += 1
    else:
        print(f"  No changes needed (already patched)")

print("\n" + "=" * 70)
if patched_count > 0:
    print(f"✓ Successfully patched {patched_count} file(s)")
else:
    print("✓ All files already patched")
print("=" * 70)

# Verify by showing one of the patched lines
print("\nVerification - checking truthx.py line 213:")
truthx_file = os.path.join(cache_dir, 'truthx.py')
if os.path.exists(truthx_file):
    with open(truthx_file, 'r') as f:
        lines = f.readlines()
        if len(lines) >= 213:
            print(f"  Line 213: {lines[212].strip()}")
            if 'weights_only=False' in lines[212]:
                print("  ✓ Patch verified!")
            else:
                print("  ⚠️  weights_only=False not found on line 213")
                print("  Let me show lines around 213:")
                for i in range(210, min(216, len(lines))):
                    marker = ">>>" if i == 212 else "   "
                    print(f"  {marker} {i+1}: {lines[i].rstrip()}")

print("\nNow run: python run_truthx.py")
print("=" * 70)