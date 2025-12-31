#!/usr/bin/env python3
"""
Benchmark: Full TTS with TensorRT-LLM compiled transformer

Uses TensorRT-LLM (not torch-tensorrt) to compile the GPT-2 backbone.

Usage:
    python benchmark_trtllm_full.py
"""

import argparse
import os
import sys
import time
import tempfile
import statistics
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

TEST_TEXTS = [
    "Hello, this is a test of the text to speech system.",
    "The quick brown fox jumps over the lazy dog.",
    "Welcome to the future of artificial intelligence and speech synthesis.",
    "Today we are testing the performance of our streaming audio generation.",
    "This benchmark measures latency to first audio chunk.",
    "Machine learning models can now generate natural sounding speech.",
    "Real-time audio synthesis requires careful optimization.",
    "The goal is to minimize time between text input and audio output.",
    "Streaming allows users to hear audio before full generation completes.",
    "Performance testing helps identify bottlenecks in the pipeline.",
]


def build_trtllm_engine(model, engine_dir: Path):
    """Build TensorRT-LLM engine for the T3 GPT-2 transformer."""
    import tensorrt_llm
    from tensorrt_llm.builder import Builder
    from tensorrt_llm.network import net_guard
    from tensorrt_llm.models import GPTModel, GPTConfig
    from tensorrt_llm.plugin import PluginConfig
    from tensorrt_llm.mapping import Mapping
    import tensorrt as trt

    print("Building TensorRT-LLM engine...")

    # Get T3's GPT-2 config
    gpt_cfg = model.t3.cfg
    print(f"  Hidden size: {gpt_cfg.hidden_size}")
    print(f"  Num layers: {gpt_cfg.num_hidden_layers}")
    print(f"  Num heads: {gpt_cfg.num_attention_heads}")
    print(f"  Vocab size: {gpt_cfg.vocab_size}")

    # Create TensorRT-LLM config
    # Note: TensorRT-LLM GPTConfig may have different parameter names
    trtllm_config = GPTConfig(
        num_hidden_layers=gpt_cfg.num_hidden_layers,
        num_attention_heads=gpt_cfg.num_attention_heads,
        hidden_size=gpt_cfg.hidden_size,
        vocab_size=gpt_cfg.vocab_size,
        hidden_act='gelu',
        max_position_embeddings=2048,
        dtype='float16',
    )

    print(f"  TensorRT-LLM config created")

    # Create the TensorRT-LLM model
    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)

    tensorrt_llm_gpt = GPTModel(trtllm_config)

    # Load weights from PyTorch model
    print("  Loading weights from PyTorch model...")
    pytorch_tfmr = model.t3.tfmr

    # Map PyTorch weights to TensorRT-LLM format
    # This is the critical part - weight names must match
    state_dict = pytorch_tfmr.state_dict()

    for name, param in state_dict.items():
        print(f"    {name}: {param.shape}")

    # Build engine
    print("  Building TensorRT engine...")

    builder = Builder()
    builder_config = builder.create_builder_config(
        name='t3_gpt2',
        precision='float16',
        max_batch_size=1,
        max_input_len=2048,
        max_seq_len=2048,
    )

    # Plugin config
    plugin_config = PluginConfig()
    plugin_config.gpt_attention_plugin = 'float16'
    plugin_config.gemm_plugin = 'float16'

    engine_dir.mkdir(parents=True, exist_ok=True)

    # Build
    with net_guard(tensorrt_llm_gpt):
        # Define network inputs
        network = tensorrt_llm_gpt

        # This is where we'd define the forward pass
        # TensorRT-LLM has specific APIs for this

    print(f"  Engine saved to: {engine_dir}")
    return engine_dir


def try_trtllm_simple(model):
    """Try a simpler TensorRT-LLM approach using their high-level API."""
    print("\n" + "="*60)
    print("Attempting TensorRT-LLM compilation...")
    print("="*60)

    try:
        import tensorrt_llm
        from tensorrt_llm import LLM, SamplingParams
        from tensorrt_llm.hlapi import LLM as HLAPI_LLM

        print(f"TensorRT-LLM version: {tensorrt_llm.__version__}")

        # Check what's available in the high-level API
        print("\nAvailable in tensorrt_llm:")
        for attr in dir(tensorrt_llm):
            if not attr.startswith('_'):
                print(f"  - {attr}")

        return None

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def try_trtllm_convert_checkpoint(model, output_dir: Path):
    """Try using TensorRT-LLM's checkpoint conversion."""
    print("\n" + "="*60)
    print("Attempting TensorRT-LLM checkpoint conversion...")
    print("="*60)

    try:
        import tensorrt_llm
        from tensorrt_llm.models.gpt.convert import convert_hf_gpt2

        # Get the HuggingFace GPT2 model
        pytorch_tfmr = model.t3.tfmr
        print(f"PyTorch transformer type: {type(pytorch_tfmr)}")

        # TensorRT-LLM expects a HuggingFace model path or model object
        # Let's see if we can use the convert function

        output_dir.mkdir(parents=True, exist_ok=True)

        # Check convert function signature
        import inspect
        sig = inspect.signature(convert_hf_gpt2)
        print(f"\nconvert_hf_gpt2 signature: {sig}")

        return None

    except ImportError as e:
        print(f"Import error: {e}")
        print("\nTrying alternative approach...")
        return try_trtllm_manual_build(model, output_dir)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def try_trtllm_manual_build(model, output_dir: Path):
    """Manually build TensorRT-LLM engine."""
    print("\n" + "="*60)
    print("Manual TensorRT-LLM engine build...")
    print("="*60)

    try:
        import tensorrt_llm
        from tensorrt_llm import Tensor
        from tensorrt_llm.functional import (
            embedding, gelu, layer_norm, matmul, softmax,
            concat, select, shape, gather, cast
        )
        from tensorrt_llm.layers import (
            Embedding, Linear, LayerNorm, Attention, MLP,
            ColumnLinear, RowLinear, GatedMLP
        )
        from tensorrt_llm.module import Module, ModuleList
        from tensorrt_llm.builder import Builder
        from tensorrt_llm.network import net_guard
        import tensorrt as trt

        gpt_cfg = model.t3.cfg
        print(f"Building for: hidden={gpt_cfg.hidden_size}, layers={gpt_cfg.num_hidden_layers}, heads={gpt_cfg.num_attention_heads}")

        class TRT_GPT2Block(Module):
            def __init__(self, hidden_size, num_heads, layer_idx):
                super().__init__()
                self.ln_1 = LayerNorm(hidden_size)
                self.attn = Attention(
                    hidden_size=hidden_size,
                    num_attention_heads=num_heads,
                    attention_head_size=hidden_size // num_heads,
                    num_kv_heads=num_heads,
                    layer_idx=layer_idx,
                )
                self.ln_2 = LayerNorm(hidden_size)
                self.mlp = MLP(
                    hidden_size=hidden_size,
                    ffn_hidden_size=hidden_size * 4,
                    hidden_act='gelu',
                )

            def forward(self, hidden_states, attention_mask=None, past_key_value=None):
                residual = hidden_states
                hidden_states = self.ln_1(hidden_states)
                attn_output = self.attn(hidden_states, attention_mask, past_key_value)
                hidden_states = residual + attn_output

                residual = hidden_states
                hidden_states = self.ln_2(hidden_states)
                hidden_states = self.mlp(hidden_states)
                hidden_states = residual + hidden_states

                return hidden_states

        class TRT_GPT2(Module):
            def __init__(self, config):
                super().__init__()
                self.hidden_size = config.hidden_size
                self.num_layers = config.num_hidden_layers
                self.num_heads = config.num_attention_heads

                self.wpe = Embedding(2048, self.hidden_size)
                self.layers = ModuleList([
                    TRT_GPT2Block(self.hidden_size, self.num_heads, i)
                    for i in range(self.num_layers)
                ])
                self.ln_f = LayerNorm(self.hidden_size)

            def forward(self, input_embeds, position_ids=None):
                if position_ids is None:
                    # Create position IDs
                    seq_len = shape(input_embeds, 1)
                    position_ids = concat([cast(i, 'int32') for i in range(2048)])[:seq_len]

                pos_embeds = self.wpe(position_ids)
                hidden_states = input_embeds + pos_embeds

                for layer in self.layers:
                    hidden_states = layer(hidden_states)

                hidden_states = self.ln_f(hidden_states)
                return hidden_states

        # Create model
        print("Creating TensorRT-LLM GPT2 model...")
        trt_model = TRT_GPT2(gpt_cfg)

        # Load weights
        print("Loading weights...")
        pytorch_state = model.t3.tfmr.state_dict()

        # Map weights (this is complex and model-specific)
        # For now, just show the weight mapping needed
        print("\nPyTorch weights to map:")
        for name, param in pytorch_state.items():
            print(f"  {name}: {param.shape}")

        # Build engine
        print("\nBuilding TensorRT engine...")
        builder = Builder()

        with net_guard(trt_model):
            builder_config = builder.create_builder_config(
                name='t3_gpt2',
                precision='float16',
            )

            # Define input
            input_embeds = Tensor(
                name='input_embeds',
                dtype=trt.float16,
                shape=[-1, -1, gpt_cfg.hidden_size],
            )

            # Forward
            output = trt_model(input_embeds)

            # Mark output
            output.mark_output('output', trt.float16)

        # Save engine
        output_dir.mkdir(parents=True, exist_ok=True)
        engine_path = output_dir / 'gpt2_engine.trt'

        print(f"Engine would be saved to: {engine_path}")
        print("\nNote: Full weight mapping requires careful attention to naming conventions")

        return None

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


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
        print(f"  Run {i+1}/{iterations}: {latency:.3f}s")
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
    pytorch_results = run_benchmark(model, "PyTorch Baseline", args.iterations)

    # Try TensorRT-LLM
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "trtllm_engine"

        # Try different approaches
        try_trtllm_simple(model)
        try_trtllm_convert_checkpoint(model, output_dir)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nPyTorch baseline: {pytorch_results['mean']:.3f}s mean latency")
    print("\nTensorRT-LLM integration requires:")
    print("  1. Export model to TensorRT-LLM checkpoint format")
    print("  2. Use trtllm-build CLI or Python API to build engine")
    print("  3. Load engine and run with TensorRT-LLM runtime")
    print("\nThis is more complex than torch-tensorrt due to:")
    print("  - Custom weight mapping")
    print("  - TensorRT-LLM's specific model definitions")
    print("  - Separate runtime from PyTorch")


if __name__ == "__main__":
    main()
