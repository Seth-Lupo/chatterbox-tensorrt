#!/usr/bin/env python3
"""
Benchmark: Full TTS with TensorRT-LLM compiled transformer

Uses TensorRT-LLM to compile the GPT-2 backbone of T3.

Usage:
    python benchmark_trtllm_full.py
"""

import argparse
import os
import sys
import time
import json
import tempfile
import statistics
import subprocess
from pathlib import Path

import torch
import numpy as np
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parent / "src"))

TEST_TEXTS = [
    "Hello, this is a test of the text to speech system.",
    "The quick brown fox jumps over the lazy dog.",
    "Welcome to the future of artificial intelligence.",
    "Today we are testing performance of audio generation.",
    "This benchmark measures latency to first audio chunk.",
    "Machine learning models generate natural speech.",
    "Real-time synthesis requires careful optimization.",
    "Minimize time between text input and audio output.",
    "Streaming allows hearing audio before generation completes.",
    "Performance testing identifies pipeline bottlenecks.",
]


def export_gpt2_for_trtllm(model, output_dir: Path):
    """Export T3's GPT-2 weights in TensorRT-LLM format."""
    print("\n" + "="*60)
    print("Exporting GPT-2 weights for TensorRT-LLM...")
    print("="*60)

    output_dir.mkdir(parents=True, exist_ok=True)

    gpt_cfg = model.t3.cfg
    pytorch_tfmr = model.t3.tfmr

    # Create TensorRT-LLM config
    config = {
        "architecture": "GPTForCausalLM",
        "dtype": "float16",
        "num_hidden_layers": gpt_cfg.num_hidden_layers,
        "num_attention_heads": gpt_cfg.num_attention_heads,
        "hidden_size": gpt_cfg.hidden_size,
        "intermediate_size": gpt_cfg.hidden_size * 4,
        "vocab_size": gpt_cfg.vocab_size,
        "max_position_embeddings": 2048,
        "hidden_act": "gelu",
        "norm_epsilon": 1e-5,
        "position_embedding_type": "learned_absolute",
        "quantization": {"quant_algo": None},
        "mapping": {"world_size": 1, "tp_size": 1, "pp_size": 1},
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved to {output_dir / 'config.json'}")

    # Convert PyTorch weights to TensorRT-LLM naming
    state_dict = pytorch_tfmr.state_dict()
    trtllm_weights = {}

    print(f"\n  Converting {len(state_dict)} weight tensors...")

    for name, param in state_dict.items():
        # TensorRT-LLM uses different naming conventions
        new_name = name

        # wpe -> transformer.position_embedding
        if name == "wpe.weight":
            new_name = "transformer.position_embedding.weight"

        # h.X -> transformer.layers.X
        elif name.startswith("h."):
            parts = name.split(".")
            layer_idx = parts[1]

            if "ln_1" in name:
                new_name = f"transformer.layers.{layer_idx}.input_layernorm.weight" if "weight" in name else f"transformer.layers.{layer_idx}.input_layernorm.bias"
            elif "ln_2" in name:
                new_name = f"transformer.layers.{layer_idx}.post_attention_layernorm.weight" if "weight" in name else f"transformer.layers.{layer_idx}.post_attention_layernorm.bias"
            elif "attn.c_attn" in name:
                # Attention QKV projection
                if "weight" in name:
                    new_name = f"transformer.layers.{layer_idx}.attention.qkv.weight"
                else:
                    new_name = f"transformer.layers.{layer_idx}.attention.qkv.bias"
            elif "attn.c_proj" in name:
                # Attention output projection
                if "weight" in name:
                    new_name = f"transformer.layers.{layer_idx}.attention.dense.weight"
                else:
                    new_name = f"transformer.layers.{layer_idx}.attention.dense.bias"
            elif "mlp.c_fc" in name:
                # MLP first layer
                if "weight" in name:
                    new_name = f"transformer.layers.{layer_idx}.mlp.fc.weight"
                else:
                    new_name = f"transformer.layers.{layer_idx}.mlp.fc.bias"
            elif "mlp.c_proj" in name:
                # MLP second layer
                if "weight" in name:
                    new_name = f"transformer.layers.{layer_idx}.mlp.proj.weight"
                else:
                    new_name = f"transformer.layers.{layer_idx}.mlp.proj.bias"

        # ln_f -> transformer.ln_f
        elif name.startswith("ln_f"):
            new_name = f"transformer.{name}"

        # Convert to half precision and contiguous
        trtllm_weights[new_name] = param.half().contiguous()

    # Save weights
    save_file(trtllm_weights, output_dir / "model.safetensors")
    print(f"  Weights saved to {output_dir / 'model.safetensors'}")
    print(f"  Total weights: {len(trtllm_weights)}")

    return output_dir


def build_trtllm_engine(checkpoint_dir: Path, engine_dir: Path):
    """Build TensorRT-LLM engine using trtllm-build CLI."""
    print("\n" + "="*60)
    print("Building TensorRT-LLM engine...")
    print("="*60)

    engine_dir.mkdir(parents=True, exist_ok=True)

    # Check if trtllm-build is available
    try:
        result = subprocess.run(["trtllm-build", "--help"], capture_output=True, text=True)
        print("  trtllm-build found")
    except FileNotFoundError:
        print("  ERROR: trtllm-build not found in PATH")
        print("  Trying Python API instead...")
        return build_trtllm_engine_python(checkpoint_dir, engine_dir)

    # Build command
    cmd = [
        "trtllm-build",
        f"--checkpoint_dir={checkpoint_dir}",
        f"--output_dir={engine_dir}",
        "--gemm_plugin=float16",
        "--gpt_attention_plugin=float16",
        "--max_batch_size=1",
        "--max_input_len=2048",
        "--max_seq_len=2048",
    ]

    print(f"  Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("  Engine built successfully!")
            return engine_dir
        else:
            print(f"  Build failed: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        print("  Build timed out (5 min limit)")
        return None
    except Exception as e:
        print(f"  Build error: {e}")
        return None


def build_trtllm_engine_python(checkpoint_dir: Path, engine_dir: Path):
    """Build engine using TensorRT-LLM Python API."""
    print("\n  Using Python API to build engine...")

    try:
        import tensorrt_llm
        from tensorrt_llm.builder import Builder
        from tensorrt_llm.models.gpt.model import GPTLMHeadModel
        from tensorrt_llm.plugin import PluginConfig

        # Load config
        with open(checkpoint_dir / "config.json") as f:
            config = json.load(f)

        print(f"  Config: layers={config['num_hidden_layers']}, hidden={config['hidden_size']}")

        # This would require implementing the full model loading
        # TensorRT-LLM's Python API is complex

        print("  Note: Full Python API build requires more setup")
        print("  Recommend using trtllm-build CLI instead")

        return None

    except Exception as e:
        print(f"  Python API error: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_trtllm_engine(engine_dir: Path):
    """Load TensorRT-LLM engine for inference."""
    print("\n" + "="*60)
    print("Loading TensorRT-LLM engine...")
    print("="*60)

    try:
        from tensorrt_llm.runtime import ModelRunner

        runner = ModelRunner.from_dir(str(engine_dir))
        print("  Engine loaded successfully!")
        return runner

    except Exception as e:
        print(f"  Load error: {e}")
        return None


class TRTLLMWrapper(torch.nn.Module):
    """Wrapper to use TensorRT-LLM engine in place of PyTorch GPT-2."""

    def __init__(self, runner, hidden_size):
        super().__init__()
        self.runner = runner
        self.hidden_size = hidden_size

    def forward(self, inputs_embeds, **kwargs):
        # TensorRT-LLM expects different input format
        # This is a simplified wrapper

        # Run inference
        with torch.no_grad():
            # The actual API depends on TensorRT-LLM version
            outputs = self.runner.generate(
                inputs_embeds,
                max_new_tokens=1,  # We just need hidden states
            )

        return outputs


def measure_time_to_first_audio(model, text: str) -> float:
    """Measure time to first audio chunk."""
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    generator = model.generate_stream(text=text, chunk_size=25, context_window=50)
    first_chunk, metrics = next(generator)
    torch.cuda.synchronize()
    latency = time.perf_counter() - start_time

    for _ in generator:
        pass

    return latency


def run_benchmark(model, name: str, iterations: int) -> dict:
    """Run benchmark."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")

    latencies = []

    print("Warmup...")
    _ = measure_time_to_first_audio(model, TEST_TEXTS[0])
    torch.cuda.empty_cache()

    for i in range(iterations):
        text = TEST_TEXTS[i % len(TEST_TEXTS)]
        latency = measure_time_to_first_audio(model, text)
        latencies.append(latency)
        print(f"  Run {i+1}/{iterations}: {latency:.3f}s - \"{text[:35]}...\"")
        torch.cuda.empty_cache()

    results = {
        "mean": statistics.mean(latencies),
        "min": min(latencies),
        "max": max(latencies),
        "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0,
    }

    print(f"\nResults: Mean={results['mean']:.3f}s, Min={results['min']:.3f}s, Max={results['max']:.3f}s")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--engine-dir", type=str, default=None, help="Pre-built engine directory")
    parser.add_argument("--export-only", action="store_true", help="Only export weights, don't benchmark")
    args = parser.parse_args()

    print("="*60)
    print("TensorRT-LLM Full Benchmark")
    print("="*60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    print("\nLoading Chatterbox model...")
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    model = ChatterboxTurboTTS.from_pretrained(
        device="cuda",
        dtype=args.dtype,
        compile_mode=None,
    )
    print("Model loaded")

    # PyTorch baseline
    if not args.export_only:
        pytorch_results = run_benchmark(model, "PyTorch Baseline", args.iterations)

    # Export and build TensorRT-LLM engine
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        checkpoint_dir = tmpdir / "checkpoint"
        engine_dir = Path(args.engine_dir) if args.engine_dir else tmpdir / "engine"

        # Export weights
        export_gpt2_for_trtllm(model, checkpoint_dir)

        if args.export_only:
            # Copy to permanent location
            import shutil
            perm_dir = Path("trtllm_checkpoint")
            if perm_dir.exists():
                shutil.rmtree(perm_dir)
            shutil.copytree(checkpoint_dir, perm_dir)
            print(f"\nCheckpoint exported to: {perm_dir}")
            print("\nTo build engine, run:")
            print(f"  trtllm-build --checkpoint_dir={perm_dir} --output_dir=trtllm_engine --gemm_plugin=float16 --gpt_attention_plugin=float16 --max_batch_size=1 --max_input_len=2048 --max_seq_len=2048")
            return

        # Build engine
        if args.engine_dir and Path(args.engine_dir).exists():
            print(f"\nUsing pre-built engine from {args.engine_dir}")
            built_engine_dir = Path(args.engine_dir)
        else:
            built_engine_dir = build_trtllm_engine(checkpoint_dir, engine_dir)

        if built_engine_dir:
            # Load and benchmark
            runner = load_trtllm_engine(built_engine_dir)
            if runner:
                print("\nTensorRT-LLM engine ready!")
                print("Note: Full integration requires replacing T3's transformer calls")
            else:
                print("\nEngine built but failed to load")
        else:
            print("\nEngine build failed")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if not args.export_only:
        print(f"\nPyTorch baseline: {pytorch_results['mean']:.3f}s mean latency")

    print("\nTensorRT-LLM Status:")
    print("  - Weights exported: YES")
    print("  - Engine built: Check above")
    print("\nTo use TensorRT-LLM in production:")
    print("  1. Export: python benchmark_trtllm_full.py --export-only")
    print("  2. Build:  trtllm-build --checkpoint_dir=trtllm_checkpoint ...")
    print("  3. Load engine and replace T3.tfmr forward pass")


if __name__ == "__main__":
    main()
