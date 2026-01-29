#!/usr/bin/env python3
"""
Fix transformers installation for TruthX - Updated Version
Installs a newer version with cache_utils support
"""

import subprocess
import sys

print("=" * 70)
print("          TruthX Transformers Installation Fix v2")
print("=" * 70)

def run_command(cmd, description):
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
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# Step 1: Uninstall existing transformers
print("\n[1/4] Uninstalling existing transformers...")
run_command(
    "pip uninstall -y transformers",
    "Removing old installation..."
)

# Step 2: Install newer compatible version
print("\n[2/4] Installing newer transformers version...")
print("Installing transformers 4.40.0 (supports cache_utils)")
success = run_command(
    "pip install transformers==4.40.0 --break-system-packages",
    "Installing transformers 4.40.0..."
)

if not success:
    print("\n⚠️  Installation with --break-system-packages failed, trying without...")
    success = run_command(
        "pip install transformers==4.40.0",
        "Installing transformers 4.40.0 (without --break-system-packages)..."
    )

# Step 3: Also ensure accelerate is installed (needed for device_map="auto")
print("\n[3/4] Installing/updating accelerate...")
run_command(
    "pip install accelerate --break-system-packages --upgrade || pip install accelerate --upgrade",
    "Installing accelerate for device mapping..."
)

# Step 4: Verify installation
print("\n[4/4] Verifying installation...")
print("-" * 70)

# Remove old transformers from path before testing
sys.path = [p for p in sys.path if 'transformers_old' not in p]

try:
    # Force reimport
    if 'transformers' in sys.modules:
        del sys.modules['transformers']
    
    import transformers
    print(f"✓ Transformers version: {transformers.__version__}")
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("✓ AutoTokenizer imported")
    print("✓ AutoModelForCausalLM imported")
    
    # Check for cache_utils
    try:
        from transformers.cache_utils import Cache, DynamicCache
        print("✓ cache_utils available (needed for TruthX)")
    except ImportError:
        print("❌ cache_utils not available - installation may have failed")
        print("   Try manually: pip install transformers==4.40.0")
        sys.exit(1)
    
    try:
        from transformers import BitsAndBytesConfig
        print("✓ BitsAndBytesConfig imported")
    except ImportError:
        print("⚠  BitsAndBytesConfig not available (will run without quantization)")
    
    print("\n✓ All core components working!")
    verification_passed = True
    
except Exception as e:
    print(f"❌ Verification failed: {e}")
    import traceback
    traceback.print_exc()
    verification_passed = False

# Final status
print("\n" + "=" * 70)

if verification_passed:
    print("✓ Installation successful!")
    print("=" * 70)
    print("\nTransformers 4.40.0 is now installed with cache_utils support.")
    print("\nNext steps:")
    print("  1. If you haven't downloaded the model yet:")
    print("     python download_truthx.py")
    print("  2. Run TruthX inference:")
    print("     python run_truthx_local.py")
    print("=" * 70)
else:
    print("❌ Installation failed - see errors above")
    print("=" * 70)
    print("\nManual fix:")
    print("  pip uninstall -y transformers")
    print("  pip install transformers==4.40.0")
    sys.exit(1)