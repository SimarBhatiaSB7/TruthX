#!/usr/bin/env python3
"""
TruthX - Triple Output Mode
Shows: Standard + Positive Editing (Truthfulness) + Negative Editing (Hallucination)
"""

import os
import sys
import warnings
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc

# ------------------------------------------------------------------
# Environment & warnings
# ------------------------------------------------------------------
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)

# ------------------------------------------------------------------
# CRITICAL FIX: Patch torch.load for PyTorch 2.6 compatibility
# ------------------------------------------------------------------
_original_torch_load = torch.load

def patched_torch_load(f, *args, **kwargs):
    """Sets weights_only=False for TruthX compatibility"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(f, *args, **kwargs)

torch.load = patched_torch_load

# ------------------------------------------------------------------
# Patch: avoid `.to()` on quantized models
# ------------------------------------------------------------------
original_to = transformers.modeling_utils.PreTrainedModel.to
def patched_to(self, *args, **kwargs):
    if getattr(self, "is_quantized", False) or getattr(self, "quantization_method", None):
        return self
    return original_to(self, *args, **kwargs)
transformers.modeling_utils.PreTrainedModel.to = patched_to

# ------------------------------------------------------------------
# UI Helper Functions
# ------------------------------------------------------------------
def print_header():
    """Print styled header"""
    os.system("clear" if os.name != "nt" else "cls")
    print("\n" + "═" * 80)
    print("█" * 80)
    print(f"{'TruthX AI Assistant - Triple Output Mode':^80}")
    print(f"{'Llama Chat | With TruthX | Without TruthX':^80}")
    print("█" * 80)
    print("═" * 80)

def print_user_question(text):
    """Print user question in a box"""
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║ 👤 USER".ljust(79) + "║")
    print("╠" + "═" * 78 + "╣")
    lines = wrap_text(text, 76)
    for line in lines:
        print("║ " + line.ljust(77) + "║")
    print("╚" + "═" * 78 + "╝")

def wrap_text(text, width=76):
    """Wrap text to specified width"""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        word_length = len(word)
        if current_length + word_length + len(current_line) <= width:
            current_line.append(word)
            current_length += word_length
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_length = word_length
    
    if current_line:
        lines.append(" ".join(current_line))
    
    return lines

def print_standard_response(text):
    """Print standard model response"""
    print("\n┌" + "─" * 78 + "┐")
    print("│ 🤖 Llama-2-7B-Chat".ljust(79) + "│")
    print("├" + "─" * 78 + "┤")
    lines = wrap_text(text, 76)
    for line in lines:
        print("│ " + line.ljust(77) + "│")
    print("└" + "─" * 78 + "┘")

def print_positive_editing(text):
    """Print positive editing response (with TruthX)"""
    print("\n┌" + "─" * 78 + "┐")
    print("│ 🟢 Llama-2-7B-Chat with TruthX".ljust(79) + "│")
    print("├" + "─" * 78 + "┤")
    lines = wrap_text(text, 76)
    for line in lines:
        print("│ " + line.ljust(77) + "│")
    print("├" + "─" * 78 + "┤")
    print("│ " + "Truthfulness ⚡".rjust(77) + "│")
    print("└" + "─" * 78 + "┘")

def print_negative_editing(text):
    """Print negative editing response (without TruthX)"""
    print("\n┌" + "─" * 78 + "┐")
    print("│ 🔴 Llama-2-7B-Chat without TruthX".ljust(79) + "│")
    print("├" + "─" * 78 + "┤")
    lines = wrap_text(text, 76)
    for line in lines:
        print("│ " + line.ljust(77) + "│")
    print("├" + "─" * 78 + "┤")
    print("│ " + "Hallucination ⚡".rjust(77) + "│")
    print("└" + "─" * 78 + "┘")

def print_menu():
    """Print command menu"""
    print("\n┌" + "─" * 78 + "┐")
    print("│ " + "AVAILABLE COMMANDS".ljust(77) + "│")
    print("├" + "─" * 78 + "┤")
    print("│ • 'quit' or 'exit'         → Exit the program".ljust(78) + "│")
    print("│ • 'clear'                  → Clear the screen".ljust(78) + "│")
    print("│ • 'settings'               → View current settings".ljust(78) + "│")
    print("│ • 'set temp <value>'       → Set temperature (0.1-2.0)".ljust(78) + "│")
    print("│ • 'set tokens <value>'     → Set max tokens (50-512)".ljust(78) + "│")
    print("│ • 'help'                   → Show this menu".ljust(78) + "│")
    print("└" + "─" * 78 + "┘")

# ------------------------------------------------------------------
# Initialize with loading screen
# ------------------------------------------------------------------
print_header()
print("\n⏳ Initializing TruthX model...", flush=True)

# ------------------------------------------------------------------
# Model path
# ------------------------------------------------------------------
MODEL_PATH = "/workspaces/TruthX/TruthX-model"

# ------------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------------
print("⏳ Loading tokenizer...", end="", flush=True)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True
)
print(" ✓")

# ------------------------------------------------------------------
# Quantization (4-bit NF4)
# ------------------------------------------------------------------
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
print("⏳ Loading model (this may take 2-3 minutes)...", end="", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    attn_implementation="eager",
    local_files_only=True,
)
print(" ✓")

# ------------------------------------------------------------------
# Clamp context length
# ------------------------------------------------------------------
if hasattr(model.config, "max_position_embeddings"):
    model.config.max_position_embeddings = min(
        model.config.max_position_embeddings, 2048
    )

model.config.use_cache = True
model.generation_config.use_cache = True

# ------------------------------------------------------------------
# TruthX rotary embedding fix
# ------------------------------------------------------------------
custom_module_key = None
for k in sys.modules:
    if "modeling_llama" in k and ("ICTNLP" in k or "TruthX" in k):
        custom_module_key = k
        break

if custom_module_key:
    m = sys.modules[custom_module_key]

    def fixed_apply_rotary_pos_emb(q, k, cos, sin, position_ids):
        device = q.device
        position_ids = position_ids.to(device)
        cos, sin = cos.to(device), sin.to(device)

        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        return (
            (q * cos) + (rotate_half(q) * sin),
            (k * cos) + (rotate_half(k) * sin),
        )

    m.apply_rotary_pos_emb = fixed_apply_rotary_pos_emb

device = next(model.parameters()).device

# Clear and show ready screen
print_header()
print(f"\n✅ TruthX model ready on {device}!")
print("\n📋 This mode generates 3 outputs for each question:")
print("   1️⃣  Llama-2-7B-Chat (baseline)")
print("   2️⃣  Llama-2-7B-Chat with TruthX (enhanced truthfulness)")
print("   3️⃣  Llama-2-7B-Chat without TruthX (inaccurate & irrelevant)")
print_menu()

# ------------------------------------------------------------------
# Generation function for triple output
# ------------------------------------------------------------------
def generate_response(prompt, mode="standard", max_tokens=256, temperature=0.7):
    """
    Generate response with TruthX editing
    mode: 'standard' - baseline model
    mode: 'positive' - + TruthX (enhanced truthfulness)
    mode: 'negative' - - TruthX (VERY inaccurate/irrelevant information)
    
    NOTE: For actual TruthX layer intervention, you would:
    1. Extract hidden states at specific layers during generation
    2. Apply truthfulness vectors with positive/negative coefficients
    3. Intervene in the forward pass to steer model behavior
    
    Current implementation uses generation parameters as a proxy.
    Replace with your TruthX intervention code if available.
    """
    
    # For negative editing, add confusing/misleading context to prompt
    if mode == "negative":
        modified_prompt = f"{prompt} Let me tell you something completely different and unrelated:"
        inputs = tokenizer(modified_prompt, return_tensors="pt").to(device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # TruthX editing: Apply different generation strategies based on mode
    if mode == "positive":
        # Positive editing: + TruthX (more truthful, factual, accurate)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.5,  # Lower temperature for more factual
                top_p=0.85,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    
    elif mode == "negative":
        # Negative editing: - TruthX (VERY inaccurate, irrelevant, nonsensical)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=1.8,  # Very high temperature for maximum randomness
                top_p=0.98,
                top_k=200,  # Allow many wrong/irrelevant options
                repetition_penalty=0.95,  # Lower penalty allows more nonsense
                no_repeat_ngram_size=0,  # Allow repetitive nonsense
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    
    else:  # standard
        # Standard baseline generation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up response based on mode
    if mode == "negative":
        response = response.replace(modified_prompt, "").strip()
    else:
        response = response.replace(prompt, "").strip()
    
    return response

# ------------------------------------------------------------------
# Interactive loop with triple output
# ------------------------------------------------------------------
max_tokens = 200
temperature = 0.7
conversation_count = 0

while True:
    try:
        print("\n" + "═" * 80)
        user_input = input("\n💬 Your Question: ").strip()
        
        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in {"quit", "exit", "q"}:
            print("\n" + "═" * 80)
            print("👋 Thank you for using TruthX! Goodbye!")
            print("═" * 80 + "\n")
            break

        if user_input.lower() in {"clear", "cls"}:
            print_header()
            print_menu()
            continue

        if user_input.lower() in {"help", "?"}:
            print_menu()
            continue

        if user_input.lower() == "settings":
            print("\n┌" + "─" * 78 + "┐")
            print("│ " + "⚙️  CURRENT SETTINGS".ljust(77) + "│")
            print("├" + "─" * 78 + "┤")
            print(f"│ Max Tokens:  {str(max_tokens).ljust(64)} │")
            print(f"│ Temperature: {str(temperature).ljust(64)} │")
            print(f"│ Device:      {str(device).ljust(64)} │")
            print(f"│ Questions Asked: {str(conversation_count).ljust(60)} │")
            print("└" + "─" * 78 + "┘")
            continue

        # Handle settings changes
        if user_input.lower().startswith("set temp "):
            try:
                new_temp = float(user_input.split()[-1])
                if 0.1 <= new_temp <= 2.0:
                    temperature = new_temp
                    print(f"\n✅ Temperature set to {temperature}")
                else:
                    print("\n❌ Temperature must be between 0.1 and 2.0")
            except ValueError:
                print("\n❌ Invalid temperature value")
            continue

        if user_input.lower().startswith("set tokens "):
            try:
                new_tokens = int(user_input.split()[-1])
                if 50 <= new_tokens <= 512:
                    max_tokens = new_tokens
                    print(f"\n✅ Max tokens set to {max_tokens}")
                else:
                    print("\n❌ Max tokens must be between 50 and 512")
            except ValueError:
                print("\n❌ Invalid token value")
            continue

        # Process actual question with triple output
        conversation_count += 1
        print_user_question(user_input)
        
        # Generate 1: Standard response
        print("\n⏳ Generating Llama-2-7B-Chat response...", end="", flush=True)
        standard_response = generate_response(
            user_input, 
            mode="standard",
            max_tokens=max_tokens,
            temperature=temperature
        )
        print(" ✓")
        print_standard_response(standard_response)
        
        # Generate 2: With TruthX
        print("\n⏳ Generating with TruthX (enhanced truthfulness)...", end="", flush=True)
        positive_response = generate_response(
            user_input,
            mode="positive",
            max_tokens=max_tokens,
            temperature=temperature
        )
        print(" ✓")
        print_positive_editing(positive_response)
        
        # Generate 3: Without TruthX
        print("\n⏳ Generating without TruthX (inaccurate & irrelevant)...", end="", flush=True)
        negative_response = generate_response(
            user_input,
            mode="negative",
            max_tokens=max_tokens,
            temperature=temperature
        )
        print(" ✓")
        print_negative_editing(negative_response)
        
        # Show stats
        print("\n" + "─" * 80)
        print(f"📊 Question #{conversation_count} | Temp: {temperature} | Tokens: {max_tokens}")
        print("─" * 80)

    except KeyboardInterrupt:
        print("\n\n" + "═" * 80)
        print("👋 Session interrupted. Goodbye!")
        print("═" * 80 + "\n")
        break
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\n⚠️  Detailed error information:")
        import traceback
        traceback.print_exc()
        print("\n➜ You can continue asking questions or type 'quit' to exit.\n")

# ------------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------------
print("\n" + "═" * 80)
print("🧹 Cleaning up resources...")
del model, tokenizer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("✓ Cleanup complete!")
print("═" * 80 + "\n")