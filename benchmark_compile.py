#!/usr/bin/env python3
"""
Benchmark: Optimized Chatterbox Turbo with torch.compile

Tests time-to-first-audio latency with fp32 and torch.compile optimization.

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

    # Warmup runs (compile happens here)
    print("Warmup runs (triggering compilation)...")
    for i in range(3):
        _ = measure_time_to_first_audio(model, TEST_TEXTS[i], audio_prompt_path)
        torch.cuda.empty_cache()
    print("Warmup complete.\n")

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

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark optimized Chatterbox Turbo")
    parser.add_argument("--iterations", type=int, default=10, help="Number of test iterations")
    parser.add_argument("--audio_prompt", type=str, default=None, help="Reference audio for voice cloning")
    parser.add_argument("--compile_mode", type=str, default="tensorrt",
                        choices=["tensorrt", "default", "max-autotune"],
                        help="Compilation mode")
    args = parser.parse_args()

    print("="*60)
    print("Chatterbox Turbo - Optimized Benchmark (FP32)")
    print("="*60)

    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

        # Compute capability
        cc = torch.cuda.get_device_capability(0)
        print(f"Compute Capability: {cc[0]}.{cc[1]}")
    else:
        print("ERROR: CUDA not available")
        sys.exit(1)

    print(f"Dtype: float32")
    print(f"Compile mode: {args.compile_mode}")
    print(f"Iterations: {args.iterations}")
    print(f"Audio prompt: {args.audio_prompt or 'None (using default voice)'}")

    # =========================================================================
    # Load optimized model
    # =========================================================================
    print("\n" + "="*60)
    print(f"Loading model with fp32 + torch.compile({args.compile_mode})...")
    print("="*60)

    load_start = time.perf_counter()
    model = ChatterboxTurboTTS.from_pretrained(
        device="cuda",
        dtype="float32",
        compile_mode=args.compile_mode,
    )
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.2f}s")

    # Run benchmark
    results = run_benchmark(
        model,
        f"FP32 + torch.compile({args.compile_mode})",
        args.iterations,
        args.audio_prompt
    )

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)

    print(f"\nConfiguration: FP32 + torch.compile({args.compile_mode})")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("-"*60)
    print(f"  Mean latency:   {results['mean']:.3f}s")
    print(f"  Min latency:    {results['min']:.3f}s")
    print(f"  Max latency:    {results['max']:.3f}s")
    print(f"  Std deviation:  {results['stdev']:.3f}s")
    print("-"*60)

    # Performance assessment
    mean_latency = results['mean']
    if mean_latency < 0.3:
        print("Performance: EXCELLENT (< 300ms first chunk)")
    elif mean_latency < 0.5:
        print("Performance: VERY GOOD (< 500ms first chunk)")
    elif mean_latency < 1.0:
        print("Performance: GOOD (< 1s first chunk)")
    else:
        print("Performance: NEEDS OPTIMIZATION (> 1s first chunk)")

    print("="*60)


if __name__ == "__main__":
    main()
