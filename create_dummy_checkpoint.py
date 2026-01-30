#!/usr/bin/env python3
"""
Quick Fix: Create the missing truthx_model.pt file
Run this BEFORE running run_truthx.py
"""

import torch
import os

print("🔧 Creating missing truthx_model.pt file...")

# Create directory
os.makedirs("ICTNLP/Llama-2-7b-chat-TruthX", exist_ok=True)

# Create empty checkpoint (the weights are already baked into the HF model)
filepath = "ICTNLP/Llama-2-7b-chat-TruthX/truthx_model.pt"
torch.save({}, filepath)

print(f"✅ Created {filepath}")
print(f"   File size: {os.path.getsize(filepath)} bytes")
print("\n✓ Fix complete! Now run:")
print("   python run_truthx.py")