#!/usr/bin/env python3
"""
Benchmark: CUDA vs torch.compile with TensorRT backend

Compares time-to-first-audio for:
1. Plain CUDA (no compilation)
2. torch.compile with TensorRT backend

Usage:
    python benchmark_compile.py
    python benchmark_compile.py --iterations 20
    python benchmark_compile.py --audio_prompt voice.wav
"""

import argparse
import time
import statistics
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chatterbox.tts_turbo import ChatterboxTurboTTS

# Test sentences of varying lengths
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


def measure_time_to_first_audio(model, text: str, audio_prompt_path: str = None) -> float:
    """Measure time from generation start to first audio chunk."""
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    generator = model.generate_stream(
        text=text,
        audio_prompt_path=audio_prompt_path,
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


def run_benchmark(model, name: str, iterations: int, audio_prompt_path: str = None) -> dict:
    """Run benchmark for a model configuration."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")

    latencies = []

    # Warmup run
    print("Warmup run...")
    _ = measure_time_to_first_audio(model, TEST_TEXTS[0], audio_prompt_path)
    torch.cuda.empty_cache()

    # Benchmark runs
    for i in range(iterations):
        text = TEST_TEXTS[i % len(TEST_TEXTS)]
        latency = measure_time_to_first_audio(model, text, audio_prompt_path)
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
        "latencies": latencies,
    }

    print(f"\nResults for {name}:")
    print(f"  Mean:   {results['mean']:.3f}s")
    print(f"  Min:    {results['min']:.3f}s")
    print(f"  Max:    {results['max']:.3f}s")
    print(f"  Stdev:  {results['stdev']:.3f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark CUDA vs torch.compile TensorRT")
    parser.add_argument("--iterations", type=int, default=10, help="Number of test iterations")
    parser.add_argument("--audio_prompt", type=str, default=None, help="Reference audio for voice cloning")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"], help="Data type")
    args = parser.parse_args()

    print("="*60)
    print("Chatterbox Turbo Compilation Benchmark")
    print("="*60)

    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("ERROR: CUDA not available")
        sys.exit(1)

    print(f"Iterations: {args.iterations}")
    print(f"Dtype: {args.dtype}")
    print(f"Audio prompt: {args.audio_prompt or 'None (using default voice)'}")

    # =========================================================================
    # Load and benchmark compiled model
    # =========================================================================
    print("\n" + "="*60)
    print("Loading model with torch.compile (dynamic=True)...")
    print("="*60)

    load_start = time.perf_counter()
    model = ChatterboxTurboTTS.from_pretrained(
        device="cuda",
        dtype=args.dtype,
        compile_mode="default",  # Uses dynamic=True internally
    )
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.2f}s")

    results = run_benchmark(
        model,
        "torch.compile (dynamic=True)",
        args.iterations,
        args.audio_prompt
    )

    # Free memory
    del model
    torch.cuda.empty_cache()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)

    print(f"\n{'Metric':<30} {'Value':<15}")
    print("-"*45)
    print(f"{'Mean latency to first chunk':<30} {results['mean']:.3f}s")
    print(f"{'Min latency':<30} {results['min']:.3f}s")
    print(f"{'Max latency':<30} {results['max']:.3f}s")
    print(f"{'Std dev':<30} {results['stdev']:.3f}s")
    print(f"{'Iterations':<30} {results['iterations']}")

    # Real-time assessment
    print("\n" + "-"*45)
    if results['mean'] < 0.5:
        print("EXCELLENT: Sub-500ms first chunk latency")
    elif results['mean'] < 1.0:
        print("GOOD: Sub-1s first chunk latency")
    else:
        print("Consider optimization for lower latency")
    print("-"*45)


if __name__ == "__main__":
    main()
