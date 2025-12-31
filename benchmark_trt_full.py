#!/usr/bin/env python3
"""
Benchmark: Full TTS generation with TensorRT-compiled transformer

Compares time-to-first-audio for:
1. PyTorch (no compilation)
2. PyTorch with TensorRT-compiled transformer

Usage:
    python benchmark_trt_full.py
    python benchmark_trt_full.py --iterations 10
"""

import argparse
import time
import statistics
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chatterbox.tts_turbo import ChatterboxTurboTTS


# Test sentences
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


class GPT2TransformerWrapper(nn.Module):
    """Wrapper that calls GPT2 transformer blocks directly with embeddings input."""

    def __init__(self, gpt2_model):
        super().__init__()
        self.wpe = gpt2_model.wpe
        self.drop = gpt2_model.drop
        self.h = gpt2_model.h
        self.ln_f = gpt2_model.ln_f

    def forward(self, inputs_embeds, **kwargs):
        # inputs_embeds: (batch, seq_len, hidden_size)
        batch_size, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device

        # Position embeddings
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_embeds = self.wpe(position_ids)

        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        # Transformer blocks
        for block in self.h:
            outputs = block(hidden_states)
            hidden_states = outputs[0]

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        # Return in format expected by T3 (BaseModelOutputWithPastAndCrossAttentions-like)
        class Output:
            def __init__(self, last_hidden_state):
                self.last_hidden_state = last_hidden_state
            def __getitem__(self, idx):
                if idx == 0:
                    return self.last_hidden_state
                return None

        return Output(hidden_states)


def compile_transformer_trt(model):
    """Compile the T3 transformer with TensorRT."""
    import torch_tensorrt

    print("Creating transformer wrapper...")
    wrapped = GPT2TransformerWrapper(model.t3.tfmr)
    wrapped = wrapped.to(device=model.device, dtype=model.dtype).eval()

    hidden_size = model.t3.cfg.hidden_size

    print("Compiling with TensorRT (this takes ~20 seconds)...")
    compiled = torch_tensorrt.compile(
        wrapped,
        inputs=[
            torch_tensorrt.Input(
                min_shape=[1, 1, hidden_size],
                opt_shape=[1, 256, hidden_size],
                max_shape=[1, 2048, hidden_size],
                dtype=torch.float16 if model.dtype == torch.float16 else torch.float32,
            )
        ],
        enabled_precisions={torch.float16} if model.dtype == torch.float16 else {torch.float32},
        truncate_long_and_double=True,
    )

    return compiled, wrapped


def measure_time_to_first_audio(model, text: str) -> float:
    """Measure time from generation start to first audio chunk."""
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    generator = model.generate_stream(
        text=text,
        chunk_size=25,
        context_window=50,
    )

    # Get first chunk and measure time
    first_chunk, metrics = next(generator)
    torch.cuda.synchronize()
    first_chunk_time = time.perf_counter() - start_time

    # Consume remaining chunks
    for _ in generator:
        pass

    return first_chunk_time


def run_benchmark(model, name: str, iterations: int) -> dict:
    """Run benchmark for a model configuration."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")

    latencies = []

    # Warmup
    print("Warmup run...")
    _ = measure_time_to_first_audio(model, TEST_TEXTS[0])
    torch.cuda.empty_cache()

    # Benchmark runs
    for i in range(iterations):
        text = TEST_TEXTS[i % len(TEST_TEXTS)]
        latency = measure_time_to_first_audio(model, text)
        latencies.append(latency)
        print(f"  Run {i+1}/{iterations}: {latency:.3f}s - \"{text[:40]}...\"")
        torch.cuda.empty_cache()

    results = {
        "name": name,
        "iterations": iterations,
        "mean": statistics.mean(latencies),
        "min": min(latencies),
        "max": max(latencies),
        "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0,
    }

    print(f"\nResults for {name}:")
    print(f"  Mean:   {results['mean']:.3f}s")
    print(f"  Min:    {results['min']:.3f}s")
    print(f"  Max:    {results['max']:.3f}s")
    print(f"  Stdev:  {results['stdev']:.3f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark TensorRT vs PyTorch for TTS")
    parser.add_argument("--iterations", type=int, default=10, help="Number of test iterations")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    args = parser.parse_args()

    print("="*60)
    print("TensorRT vs PyTorch Full TTS Benchmark")
    print("="*60)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
    else:
        print("ERROR: CUDA not available")
        sys.exit(1)

    print(f"Iterations: {args.iterations}")
    print(f"Dtype: {args.dtype}")

    # =========================================================================
    # Test 1: PyTorch baseline (no compilation)
    # =========================================================================
    print("\n" + "="*60)
    print("Loading model (PyTorch baseline)...")
    print("="*60)

    model = ChatterboxTurboTTS.from_pretrained(
        device="cuda",
        dtype=args.dtype,
        compile_mode=None,
    )
    print("Model loaded")

    # Run PyTorch baseline
    pytorch_results = run_benchmark(model, "PyTorch (no compilation)", args.iterations)

    # =========================================================================
    # Test 2: TensorRT-compiled transformer
    # =========================================================================
    print("\n" + "="*60)
    print("Compiling transformer with TensorRT...")
    print("="*60)

    try:
        compiled_tfmr, wrapped_tfmr = compile_transformer_trt(model)
        print("Compilation successful!")

        # Swap in the compiled transformer
        # We need to monkey-patch T3 to use our wrapper
        original_tfmr = model.t3.tfmr
        model.t3.tfmr = compiled_tfmr

        # Run TensorRT benchmark
        trt_results = run_benchmark(model, "TensorRT-compiled transformer", args.iterations)

        # Restore original
        model.t3.tfmr = original_tfmr

    except Exception as e:
        print(f"TensorRT compilation failed: {e}")
        import traceback
        traceback.print_exc()
        trt_results = None

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)

    print(f"\n{'Configuration':<35} {'Mean':<10} {'Min':<10} {'Max':<10}")
    print("-"*65)
    print(f"{'PyTorch (baseline)':<35} {pytorch_results['mean']:.3f}s    {pytorch_results['min']:.3f}s    {pytorch_results['max']:.3f}s")

    if trt_results:
        print(f"{'TensorRT transformer':<35} {trt_results['mean']:.3f}s    {trt_results['min']:.3f}s    {trt_results['max']:.3f}s")

        # Speedup
        print("\n" + "-"*65)
        if trt_results['mean'] < pytorch_results['mean']:
            speedup = pytorch_results['mean'] / trt_results['mean']
            improvement = (1 - trt_results['mean'] / pytorch_results['mean']) * 100
            print(f"TensorRT is {speedup:.2f}x FASTER ({improvement:.1f}% improvement)")
        else:
            slowdown = trt_results['mean'] / pytorch_results['mean']
            print(f"TensorRT is {slowdown:.2f}x SLOWER (overhead not worth it for this workload)")
    else:
        print("\nTensorRT compilation failed - no comparison available")

    print("-"*65)


if __name__ == "__main__":
    main()
