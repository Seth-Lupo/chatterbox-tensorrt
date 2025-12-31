#!/bin/bash
#
# Build TensorRT-LLM engine for T3's GPT-2 transformer
#
# Prerequisites:
#   - TensorRT-LLM installed (pip install tensorrt-llm)
#   - CUDA toolkit
#   - T3 model checkpoint available
#
# Usage:
#   ./build_trtllm_engine.sh [--output-dir ./trtllm_engines]
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${1:-$SCRIPT_DIR/engines}"
CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints"
HF_CHECKPOINT_DIR="$CHECKPOINT_DIR/t3_gpt2_hf"
TRTLLM_CHECKPOINT_DIR="$CHECKPOINT_DIR/t3_gpt2_trtllm"
ENGINE_DIR="$OUTPUT_DIR/t3_gpt2_engine"

# T3 GPT-2 config (from llama_configs.py GPT2_medium)
HIDDEN_SIZE=1024
NUM_LAYERS=24
NUM_HEADS=16
VOCAB_SIZE=6563  # speech tokens
MAX_SEQ_LEN=2048
DTYPE="float16"

echo "============================================================"
echo "TensorRT-LLM Engine Builder for T3 GPT-2"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Hidden size:    $HIDDEN_SIZE"
echo "  Num layers:     $NUM_LAYERS"
echo "  Num heads:      $NUM_HEADS"
echo "  Vocab size:     $VOCAB_SIZE"
echo "  Max seq len:    $MAX_SEQ_LEN"
echo "  Dtype:          $DTYPE"
echo ""
echo "Directories:"
echo "  Project root:   $PROJECT_ROOT"
echo "  HF checkpoint:  $HF_CHECKPOINT_DIR"
echo "  TRTLLM ckpt:    $TRTLLM_CHECKPOINT_DIR"
echo "  Engine output:  $ENGINE_DIR"
echo ""

# Create directories
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$OUTPUT_DIR"

# Step 1: Export T3's GPT-2 weights to HuggingFace format
echo "============================================================"
echo "Step 1: Exporting T3 GPT-2 weights to HuggingFace format"
echo "============================================================"

python3 << 'PYTHON_EXPORT'
import sys
import json
from pathlib import Path

# Add project to path
project_root = Path("${PROJECT_ROOT}").resolve()
sys.path.insert(0, str(project_root / "src"))

import torch
from transformers import GPT2Config

from chatterbox.tts_turbo import ChatterboxTurboTTS

output_dir = Path("${HF_CHECKPOINT_DIR}")
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading T3 model...")
model = ChatterboxTurboTTS.from_pretrained(
    device="cuda",
    dtype="float16",
    compile_mode=None,
)

# Get the GPT-2 transformer
gpt2 = model.t3.tfmr
cfg = model.t3.cfg

print(f"T3 GPT-2 config:")
print(f"  hidden_size: {cfg.hidden_size}")
print(f"  num_hidden_layers: {cfg.num_hidden_layers}")
print(f"  num_attention_heads: {cfg.num_attention_heads}")
print(f"  vocab_size: {cfg.vocab_size}")

# Save the model and config
print(f"\nSaving to {output_dir}...")
gpt2.save_pretrained(output_dir)

# Also save the config as JSON for TensorRT-LLM
config_dict = {
    "architecture": "GPT2Model",
    "hidden_size": cfg.hidden_size,
    "num_hidden_layers": cfg.num_hidden_layers,
    "num_attention_heads": cfg.num_attention_heads,
    "vocab_size": cfg.vocab_size,
    "n_positions": cfg.n_positions,
    "n_embd": cfg.n_embd,
    "n_layer": cfg.n_layer,
    "n_head": cfg.n_head,
    "intermediate_size": cfg.n_inner if cfg.n_inner else cfg.hidden_size * 4,
    "hidden_act": "gelu_new",
    "dtype": "float16",
}

with open(output_dir / "t3_config.json", "w") as f:
    json.dump(config_dict, f, indent=2)

print("Export complete!")

# List saved files
print("\nSaved files:")
for f in output_dir.iterdir():
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"  {f.name}: {size_mb:.2f} MB")
PYTHON_EXPORT

echo ""
echo "Step 1 complete!"

# Step 2: Convert to TensorRT-LLM checkpoint format
echo ""
echo "============================================================"
echo "Step 2: Converting to TensorRT-LLM checkpoint format"
echo "============================================================"

python3 << 'PYTHON_CONVERT'
import sys
import json
import shutil
from pathlib import Path

import torch
import numpy as np

hf_dir = Path("${HF_CHECKPOINT_DIR}")
trtllm_dir = Path("${TRTLLM_CHECKPOINT_DIR}")
trtllm_dir.mkdir(parents=True, exist_ok=True)

print(f"Converting from {hf_dir} to {trtllm_dir}")

# Load HF config
with open(hf_dir / "t3_config.json") as f:
    config = json.load(f)

# Load HF weights
print("Loading HuggingFace weights...")
hf_weights = torch.load(hf_dir / "model.safetensors" if (hf_dir / "model.safetensors").exists()
                        else hf_dir / "pytorch_model.bin",
                        map_location="cpu", weights_only=True)

# TensorRT-LLM expects specific weight naming
# HF GPT-2 naming -> TensorRT-LLM naming
#
# HF: h.{layer}.attn.c_attn.weight -> TRT: transformer.layers.{layer}.attention.qkv.weight
# HF: h.{layer}.attn.c_proj.weight -> TRT: transformer.layers.{layer}.attention.dense.weight
# HF: h.{layer}.ln_1.weight -> TRT: transformer.layers.{layer}.input_layernorm.weight
# HF: h.{layer}.ln_2.weight -> TRT: transformer.layers.{layer}.post_layernorm.weight
# HF: h.{layer}.mlp.c_fc.weight -> TRT: transformer.layers.{layer}.mlp.fc.weight
# HF: h.{layer}.mlp.c_proj.weight -> TRT: transformer.layers.{layer}.mlp.proj.weight
# HF: wpe.weight -> TRT: transformer.position_embedding.weight
# HF: ln_f.weight -> TRT: transformer.ln_f.weight

print("\nWeight mapping:")
trtllm_weights = {}

for name, tensor in hf_weights.items():
    # Convert to numpy for TensorRT-LLM
    np_tensor = tensor.cpu().numpy()

    # Skip wte (token embedding) - T3 doesn't use it
    if name.startswith("wte"):
        print(f"  SKIP: {name} (T3 uses custom speech_emb)")
        continue

    # Position embeddings
    if name == "wpe.weight":
        trt_name = "transformer.position_embedding.weight"
        trtllm_weights[trt_name] = np_tensor
        print(f"  {name} -> {trt_name} {np_tensor.shape}")
        continue

    # Final layer norm
    if name == "ln_f.weight":
        trtllm_weights["transformer.ln_f.weight"] = np_tensor
        print(f"  {name} -> transformer.ln_f.weight {np_tensor.shape}")
        continue
    if name == "ln_f.bias":
        trtllm_weights["transformer.ln_f.bias"] = np_tensor
        print(f"  {name} -> transformer.ln_f.bias {np_tensor.shape}")
        continue

    # Layer weights
    if name.startswith("h."):
        parts = name.split(".")
        layer_idx = int(parts[1])

        # Attention
        if "attn.c_attn.weight" in name:
            # QKV combined weight - needs to stay combined for TRT-LLM
            trt_name = f"transformer.layers.{layer_idx}.attention.qkv.weight"
            # GPT-2 uses Conv1D which is transposed
            trtllm_weights[trt_name] = np_tensor.T
            print(f"  {name} -> {trt_name} {np_tensor.T.shape}")
        elif "attn.c_attn.bias" in name:
            trt_name = f"transformer.layers.{layer_idx}.attention.qkv.bias"
            trtllm_weights[trt_name] = np_tensor
            print(f"  {name} -> {trt_name} {np_tensor.shape}")
        elif "attn.c_proj.weight" in name:
            trt_name = f"transformer.layers.{layer_idx}.attention.dense.weight"
            trtllm_weights[trt_name] = np_tensor.T
            print(f"  {name} -> {trt_name} {np_tensor.T.shape}")
        elif "attn.c_proj.bias" in name:
            trt_name = f"transformer.layers.{layer_idx}.attention.dense.bias"
            trtllm_weights[trt_name] = np_tensor
            print(f"  {name} -> {trt_name} {np_tensor.shape}")

        # Layer norms
        elif "ln_1.weight" in name:
            trt_name = f"transformer.layers.{layer_idx}.input_layernorm.weight"
            trtllm_weights[trt_name] = np_tensor
            print(f"  {name} -> {trt_name} {np_tensor.shape}")
        elif "ln_1.bias" in name:
            trt_name = f"transformer.layers.{layer_idx}.input_layernorm.bias"
            trtllm_weights[trt_name] = np_tensor
            print(f"  {name} -> {trt_name} {np_tensor.shape}")
        elif "ln_2.weight" in name:
            trt_name = f"transformer.layers.{layer_idx}.post_layernorm.weight"
            trtllm_weights[trt_name] = np_tensor
            print(f"  {name} -> {trt_name} {np_tensor.shape}")
        elif "ln_2.bias" in name:
            trt_name = f"transformer.layers.{layer_idx}.post_layernorm.bias"
            trtllm_weights[trt_name] = np_tensor
            print(f"  {name} -> {trt_name} {np_tensor.shape}")

        # MLP
        elif "mlp.c_fc.weight" in name:
            trt_name = f"transformer.layers.{layer_idx}.mlp.fc.weight"
            trtllm_weights[trt_name] = np_tensor.T
            print(f"  {name} -> {trt_name} {np_tensor.T.shape}")
        elif "mlp.c_fc.bias" in name:
            trt_name = f"transformer.layers.{layer_idx}.mlp.fc.bias"
            trtllm_weights[trt_name] = np_tensor
            print(f"  {name} -> {trt_name} {np_tensor.shape}")
        elif "mlp.c_proj.weight" in name:
            trt_name = f"transformer.layers.{layer_idx}.mlp.proj.weight"
            trtllm_weights[trt_name] = np_tensor.T
            print(f"  {name} -> {trt_name} {np_tensor.T.shape}")
        elif "mlp.c_proj.bias" in name:
            trt_name = f"transformer.layers.{layer_idx}.mlp.proj.bias"
            trtllm_weights[trt_name] = np_tensor
            print(f"  {name} -> {trt_name} {np_tensor.shape}")
        else:
            print(f"  UNKNOWN: {name}")

# Save TensorRT-LLM checkpoint
print(f"\nSaving TensorRT-LLM checkpoint to {trtllm_dir}...")

# Save weights as safetensors or numpy
for name, weight in trtllm_weights.items():
    # Save as numpy files (TRT-LLM can load these)
    np.save(trtllm_dir / f"{name.replace('.', '_')}.npy", weight)

# Save config for TensorRT-LLM
trtllm_config = {
    "builder_config": {
        "name": "t3_gpt2",
        "precision": "float16",
        "tensor_parallel": 1,
        "pipeline_parallel": 1,
        "vocab_size": config["vocab_size"],
        "hidden_size": config["hidden_size"],
        "num_hidden_layers": config["num_hidden_layers"],
        "num_attention_heads": config["num_attention_heads"],
        "hidden_act": "gelu",
        "max_position_embeddings": config["n_positions"],
        "max_batch_size": 1,
        "max_input_len": 1024,
        "max_output_len": 1024,
    },
    "plugin_config": {
        "gpt_attention_plugin": "float16",
        "gemm_plugin": "float16",
        "context_fmha": True,
        "paged_kv_cache": False,  # Simpler for PoC
    }
}

with open(trtllm_dir / "config.json", "w") as f:
    json.dump(trtllm_config, f, indent=2)

print("Conversion complete!")
print(f"\nTotal weights: {len(trtllm_weights)}")
PYTHON_CONVERT

echo ""
echo "Step 2 complete!"

# Step 3: Build TensorRT-LLM engine
echo ""
echo "============================================================"
echo "Step 3: Building TensorRT-LLM engine"
echo "============================================================"

# Check if trtllm-build is available
if command -v trtllm-build &> /dev/null; then
    echo "Using trtllm-build CLI..."

    trtllm-build \
        --checkpoint_dir "$TRTLLM_CHECKPOINT_DIR" \
        --output_dir "$ENGINE_DIR" \
        --dtype float16 \
        --gemm_plugin float16 \
        --gpt_attention_plugin float16 \
        --max_batch_size 1 \
        --max_input_len 1024 \
        --max_output_len 1024 \
        --remove_input_padding disable \
        --paged_kv_cache disable

    echo "Engine built successfully!"
else
    echo "trtllm-build not found, using Python API..."

    python3 << 'PYTHON_BUILD'
import sys
from pathlib import Path

try:
    import tensorrt_llm
    from tensorrt_llm.builder import Builder
    from tensorrt_llm.models import GPTLMHeadModel
    from tensorrt_llm.network import net_guard
    from tensorrt_llm.plugin import PluginConfig
    import tensorrt as trt

    print(f"TensorRT-LLM version: {tensorrt_llm.__version__}")

    checkpoint_dir = Path("${TRTLLM_CHECKPOINT_DIR}")
    engine_dir = Path("${ENGINE_DIR}")
    engine_dir.mkdir(parents=True, exist_ok=True)

    # Note: Full engine building requires more setup
    # This is a placeholder showing the API
    print("\nNote: Full TensorRT-LLM engine building requires:")
    print("  1. Proper checkpoint format with rank files")
    print("  2. trtllm-build CLI (recommended)")
    print("  3. Or manual network definition with Python API")
    print("\nFor production, use: trtllm-build --checkpoint_dir <path> --output_dir <path>")

except ImportError as e:
    print(f"TensorRT-LLM not available: {e}")
    print("\nInstall with: pip install tensorrt-llm")
    print("Or use NVIDIA's container: nvcr.io/nvidia/tritonserver:XX.XX-trtllm-python-py3")
PYTHON_BUILD
fi

echo ""
echo "============================================================"
echo "Build Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Run the test script: python3 test_trtllm_t3.py"
echo "  2. Deploy with Triton: See triton_config/ directory"
echo ""
