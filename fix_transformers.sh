#!/bin/bash

echo "======================================================================"
echo "          TruthX Transformers Installation Fix"
echo "======================================================================"

echo ""
echo "[1/4] Uninstalling existing transformers..."
pip uninstall -y transformers

echo ""
echo "[2/4] Installing compatible transformers version..."
# Install a specific stable version that works with Llama-2
pip install transformers==4.35.2 --break-system-packages

echo ""
echo "[3/4] Verifying installation..."
python3 - << 'PYTHON'
import sys
sys.path = [p for p in sys.path if 'transformers_old' not in p]

try:
    import transformers
    print(f"✓ Transformers version: {transformers.__version__}")
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("✓ AutoTokenizer imported")
    print("✓ AutoModelForCausalLM imported")
    
    try:
        from transformers import BitsAndBytesConfig
        print("✓ BitsAndBytesConfig imported")
    except:
        print("⚠ BitsAndBytesConfig not available (will try without quantization)")
    
    print("\nAll core components working!")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
PYTHON

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✓ Installation successful!"
    echo "======================================================================"
    echo ""
    echo "Next step: Run the model with:"
    echo "  python run_truthx_fixed.py"
    echo ""
else
    echo ""
    echo "======================================================================"
    echo "❌ Installation failed - see errors above"
    echo "======================================================================"
    exit 1
fi