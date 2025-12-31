#!/usr/bin/env python3
"""
Convert T3 GPT-2 checkpoint and build TensorRT-LLM engine.

Compatible with TensorRT-LLM 1.1.0 and newer versions.

Usage:
    python convert_and_build.py
    python convert_and_build.py --build-only  # Skip conversion, just build
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
HF_CHECKPOINT = CHECKPOINT_DIR / "t3_gpt2_hf"
TRTLLM_CHECKPOINT = CHECKPOINT_DIR / "t3_gpt2_trtllm"
ENGINE_DIR = SCRIPT_DIR / "engines" / "t3_gpt2"


def get_trtllm_version():
    """Get TensorRT-LLM version."""
    try:
        import tensorrt_llm
        return tensorrt_llm.__version__
    except ImportError:
        return None


def convert_hf_to_trtllm():
    """Convert HuggingFace GPT-2 checkpoint to TensorRT-LLM format."""
    print("=" * 60)
    print("Converting HuggingFace checkpoint to TensorRT-LLM format")
    print("=" * 60)

    if not HF_CHECKPOINT.exists():
        print(f"ERROR: HuggingFace checkpoint not found at {HF_CHECKPOINT}")
        print("Run ./run_trtllm_poc.sh first to export the checkpoint")
        return False

    TRTLLM_CHECKPOINT.mkdir(parents=True, exist_ok=True)

    # Load HF checkpoint
    print(f"\nLoading from {HF_CHECKPOINT}...")

    # Try different file formats
    if (HF_CHECKPOINT / "model.safetensors").exists():
        from safetensors.torch import load_file
        hf_weights = load_file(HF_CHECKPOINT / "model.safetensors")
    elif (HF_CHECKPOINT / "pytorch_model.bin").exists():
        hf_weights = torch.load(
            HF_CHECKPOINT / "pytorch_model.bin",
            map_location="cpu",
            weights_only=True
        )
    else:
        print("ERROR: No model weights found")
        return False

    # Load config
    with open(HF_CHECKPOINT / "config.json") as f:
        hf_config = json.load(f)

    print(f"Model config:")
    print(f"  hidden_size: {hf_config['n_embd']}")
    print(f"  num_layers: {hf_config['n_layer']}")
    print(f"  num_heads: {hf_config['n_head']}")
    print(f"  vocab_size: {hf_config['vocab_size']}")

    # Convert weights to TensorRT-LLM format
    # TensorRT-LLM 1.x uses a specific weight naming convention
    print("\nConverting weights...")

    trtllm_weights = {}
    num_layers = hf_config['n_layer']
    hidden_size = hf_config['n_embd']
    num_heads = hf_config['n_head']
    head_size = hidden_size // num_heads

    for name, tensor in hf_weights.items():
        weight = tensor.cpu().float().numpy()

        # Position embeddings
        if name == "wpe.weight":
            trtllm_weights["transformer.position_embedding.weight"] = weight
            print(f"  {name} -> transformer.position_embedding.weight")
            continue

        # Skip token embeddings (T3 doesn't use them)
        if name == "wte.weight":
            print(f"  {name} -> SKIPPED (T3 uses custom embeddings)")
            continue

        # Final layer norm
        if name == "ln_f.weight":
            trtllm_weights["transformer.ln_f.weight"] = weight
            print(f"  {name} -> transformer.ln_f.weight")
            continue
        if name == "ln_f.bias":
            trtllm_weights["transformer.ln_f.bias"] = weight
            print(f"  {name} -> transformer.ln_f.bias")
            continue

        # Layer weights
        if name.startswith("h."):
            parts = name.split(".")
            layer_idx = int(parts[1])
            layer_prefix = f"transformer.layers.{layer_idx}"

            # Attention QKV (combined in GPT-2)
            if ".attn.c_attn.weight" in name:
                # GPT-2 uses Conv1D (transposed) and combines Q, K, V
                # Shape: (hidden_size, 3 * hidden_size)
                weight_t = weight.T  # Transpose Conv1D

                # Split into Q, K, V
                qkv_size = hidden_size
                q = weight_t[:, :qkv_size]
                k = weight_t[:, qkv_size:2*qkv_size]
                v = weight_t[:, 2*qkv_size:]

                # TensorRT-LLM expects separate or interleaved Q, K, V
                # For 1.1.0, use combined QKV
                trtllm_weights[f"{layer_prefix}.attention.qkv.weight"] = weight_t
                print(f"  {name} -> {layer_prefix}.attention.qkv.weight")

            elif ".attn.c_attn.bias" in name:
                trtllm_weights[f"{layer_prefix}.attention.qkv.bias"] = weight
                print(f"  {name} -> {layer_prefix}.attention.qkv.bias")

            elif ".attn.c_proj.weight" in name:
                trtllm_weights[f"{layer_prefix}.attention.dense.weight"] = weight.T
                print(f"  {name} -> {layer_prefix}.attention.dense.weight")

            elif ".attn.c_proj.bias" in name:
                trtllm_weights[f"{layer_prefix}.attention.dense.bias"] = weight
                print(f"  {name} -> {layer_prefix}.attention.dense.bias")

            # Layer norms
            elif ".ln_1.weight" in name:
                trtllm_weights[f"{layer_prefix}.input_layernorm.weight"] = weight
                print(f"  {name} -> {layer_prefix}.input_layernorm.weight")

            elif ".ln_1.bias" in name:
                trtllm_weights[f"{layer_prefix}.input_layernorm.bias"] = weight
                print(f"  {name} -> {layer_prefix}.input_layernorm.bias")

            elif ".ln_2.weight" in name:
                trtllm_weights[f"{layer_prefix}.post_layernorm.weight"] = weight
                print(f"  {name} -> {layer_prefix}.post_layernorm.weight")

            elif ".ln_2.bias" in name:
                trtllm_weights[f"{layer_prefix}.post_layernorm.bias"] = weight
                print(f"  {name} -> {layer_prefix}.post_layernorm.bias")

            # MLP
            elif ".mlp.c_fc.weight" in name:
                trtllm_weights[f"{layer_prefix}.mlp.fc.weight"] = weight.T
                print(f"  {name} -> {layer_prefix}.mlp.fc.weight")

            elif ".mlp.c_fc.bias" in name:
                trtllm_weights[f"{layer_prefix}.mlp.fc.bias"] = weight
                print(f"  {name} -> {layer_prefix}.mlp.fc.bias")

            elif ".mlp.c_proj.weight" in name:
                trtllm_weights[f"{layer_prefix}.mlp.proj.weight"] = weight.T
                print(f"  {name} -> {layer_prefix}.mlp.proj.weight")

            elif ".mlp.c_proj.bias" in name:
                trtllm_weights[f"{layer_prefix}.mlp.proj.bias"] = weight
                print(f"  {name} -> {layer_prefix}.mlp.proj.bias")

    # Save as safetensors (TensorRT-LLM format)
    print(f"\nSaving to {TRTLLM_CHECKPOINT}...")

    # Convert all weights to torch tensors in float16 (must be contiguous)
    torch_weights = {}
    for name, weight in trtllm_weights.items():
        torch_weights[name] = torch.from_numpy(weight.astype(np.float16)).contiguous()

    # Save as safetensors
    try:
        from safetensors.torch import save_file
        save_file(torch_weights, TRTLLM_CHECKPOINT / "rank0.safetensors")
        print(f"  Saved rank0.safetensors")
    except ImportError:
        # Fallback to PyTorch format
        torch.save(torch_weights, TRTLLM_CHECKPOINT / "rank0.bin")
        print(f"  Saved rank0.bin (safetensors not available)")

    # Save config for TensorRT-LLM (must match expected schema)
    trtllm_config = {
        "architecture": "GPTForCausalLM",
        "dtype": "float16",
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_heads,  # GPT-2 uses MHA, not GQA
        "hidden_size": hidden_size,
        "intermediate_size": hf_config.get("n_inner") or hidden_size * 4,
        "vocab_size": hf_config["vocab_size"],
        "position_embedding_type": "learned_absolute",
        "max_position_embeddings": hf_config.get("n_positions", 2048),
        "hidden_act": "gelu",
        "norm_epsilon": 1e-5,
        "quantization": {
            "quant_algo": None,
            "kv_cache_quant_algo": None,
        },
        "mapping": {
            "world_size": 1,
            "tp_size": 1,
            "pp_size": 1,
        },
    }

    with open(TRTLLM_CHECKPOINT / "config.json", "w") as f:
        json.dump(trtllm_config, f, indent=2)

    print(f"\nConversion complete!")
    print(f"  Weights: {len(trtllm_weights)} tensors")
    print(f"  Output: {TRTLLM_CHECKPOINT}")

    return True


def build_engine():
    """Build TensorRT-LLM engine."""
    print("\n" + "=" * 60)
    print("Building TensorRT-LLM Engine")
    print("=" * 60)

    version = get_trtllm_version()
    print(f"TensorRT-LLM version: {version}")

    if not TRTLLM_CHECKPOINT.exists():
        print(f"ERROR: TensorRT-LLM checkpoint not found at {TRTLLM_CHECKPOINT}")
        print("Run conversion first")
        return False

    ENGINE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import tensorrt_llm
        from tensorrt_llm.builder import Builder
        from tensorrt_llm.network import net_guard
        from tensorrt_llm.plugin import PluginConfig
        import tensorrt as trt

        # Load config
        with open(TRTLLM_CHECKPOINT / "config.json") as f:
            config = json.load(f)

        print(f"\nBuilding engine for:")
        print(f"  Hidden size: {config['hidden_size']}")
        print(f"  Num layers: {config['num_hidden_layers']}")
        print(f"  Num heads: {config['num_attention_heads']}")

        # For TensorRT-LLM 1.1.0, we need to use the Python builder API
        # This is more complex and version-specific

        print("\n" + "-" * 40)
        print("NOTE: TensorRT-LLM 1.1.0 engine building")
        print("-" * 40)
        print("""
For TensorRT-LLM 1.1.0, use the example scripts:

    cd /path/to/tensorrt_llm/examples/gpt

    python3 build.py \\
        --model_dir {checkpoint_dir} \\
        --output_dir {engine_dir} \\
        --dtype float16 \\
        --use_gpt_attention_plugin float16 \\
        --use_gemm_plugin float16 \\
        --max_batch_size 1 \\
        --max_input_len 1024 \\
        --max_output_len 1024

Or upgrade to TensorRT-LLM 0.8+ and use trtllm-build CLI.
""".format(
            checkpoint_dir=TRTLLM_CHECKPOINT,
            engine_dir=ENGINE_DIR,
        ))

        # Try to find and use the GPT example
        try:
            from tensorrt_llm.models import GPTLMHeadModel

            print("\nAttempting direct build with GPTLMHeadModel...")

            # Create model
            # This is simplified - full implementation needs weight loading
            print("Note: Direct engine building requires version-specific code.")
            print("See TensorRT-LLM examples for your version.")

        except ImportError as e:
            print(f"GPTLMHeadModel not available: {e}")

        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_next_steps():
    """Show next steps for using the converted model."""
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)

    version = get_trtllm_version()

    print(f"""
TensorRT-LLM version: {version}

Checkpoint exported to: {TRTLLM_CHECKPOINT}

For TensorRT-LLM 1.1.0, you have two options:

Option 1: Use TensorRT-LLM example scripts
-------------------------------------------
    # Clone TensorRT-LLM repo
    git clone https://github.com/NVIDIA/TensorRT-LLM.git
    cd TensorRT-LLM/examples/gpt

    # Build engine
    python3 build.py \\
        --model_dir {trtllm_ckpt} \\
        --output_dir {engine_dir} \\
        --dtype float16 \\
        --use_gpt_attention_plugin float16 \\
        --max_batch_size 1 \\
        --max_input_len 1024

Option 2: Upgrade TensorRT-LLM
------------------------------
    pip install tensorrt-llm --upgrade

    # Then use the CLI:
    trtllm-build \\
        --checkpoint_dir {trtllm_ckpt} \\
        --output_dir {engine_dir} \\
        --gemm_plugin float16

Option 3: Use PyTorch KV Cache (Already Working!)
--------------------------------------------------
    # The PyTorch implementation with KV cache is already fast
    # See benchmark results from run_trtllm_poc.sh

    output = model.t3.tfmr(
        inputs_embeds=embeds,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )
    cache = output.past_key_values

For T3 specifically, the main challenge is that it uses EMBEDDINGS
as input (not token IDs), which requires custom handling in TensorRT-LLM.
The PyTorch KV cache approach may be the best balance of performance
and simplicity for this use case.
""".format(
        trtllm_ckpt=TRTLLM_CHECKPOINT,
        engine_dir=ENGINE_DIR,
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--convert-only", action="store_true",
                        help="Only convert, don't build engine")
    parser.add_argument("--build-only", action="store_true",
                        help="Only build engine (checkpoint must exist)")
    args = parser.parse_args()

    print("=" * 60)
    print("TensorRT-LLM Conversion and Build")
    print("=" * 60)

    version = get_trtllm_version()
    if version is None:
        print("ERROR: TensorRT-LLM not installed")
        return

    print(f"TensorRT-LLM version: {version}")

    if not args.build_only:
        success = convert_hf_to_trtllm()
        if not success:
            return

    if not args.convert_only:
        build_engine()

    show_next_steps()


if __name__ == "__main__":
    main()
