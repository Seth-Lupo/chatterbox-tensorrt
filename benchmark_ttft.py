"""
Benchmark: Time to First Token (TTFT)

Measures latency to first audio chunk with different configurations:
- Full GPU (default)
- T3 on CPU, S3Gen on GPU (hybrid)
- Full CPU

Usage:
    python benchmark_ttft.py
    python benchmark_ttft.py --mode cpu
    python benchmark_ttft.py --mode hybrid
"""

import sys
sys.path.insert(0, "src")

import argparse
import time
import torch
import numpy as np
from pathlib import Path

from chatterbox.tts_turbo import (
    ChatterboxTurboTTS,
    DEFAULT_RAMP_SCHEDULE,
    normalize_text,
)


def load_model_standard(device: str) -> ChatterboxTurboTTS:
    """Load model with all components on same device."""
    print(f"Loading model on {device}...")
    model = ChatterboxTurboTTS.from_pretrained(device=device)
    print("Compiling...")
    model.compile()
    return model


def load_model_hybrid() -> ChatterboxTurboTTS:
    """Load model with T3 on CPU, S3Gen on GPU."""
    print("Loading model in HYBRID mode (T3=CPU, S3Gen=GPU)...")

    # Load on GPU first
    model = ChatterboxTurboTTS.from_pretrained(device="cuda")

    # Move T3 to CPU
    print("Moving T3 to CPU...")
    model.t3 = model.t3.to("cpu")

    # Keep S3Gen on GPU (already there)
    print("S3Gen stays on GPU")

    # Compile S3Gen (T3 on CPU doesn't benefit from compile)
    print("Compiling S3Gen...")
    model.compile()

    return model


def warmup(model: ChatterboxTurboTTS, voice_path: str, schedules: list[list[tuple]]):
    """Warmup the model with all schedule configurations."""
    print("Warming up...")
    model.prepare_conditionals(voice_path, exaggeration=0.5)

    # Warmup each schedule configuration to trigger JIT compilation
    for i, schedule in enumerate(schedules):
        print(f"  Warmup config {i+1}/{len(schedules)}: {schedule[0]} (first chunk)")
        for _ in model.generate_stream("Hello.", ramp_schedule=schedule):
            break

    print("Warmup complete")


def benchmark_ttft(
    model: ChatterboxTurboTTS,
    text: str,
    num_runs: int = 5,
    ramp_schedule=None,
) -> dict:
    """
    Benchmark time to first token/chunk.

    Returns dict with timing breakdown.
    """
    if ramp_schedule is None:
        ramp_schedule = DEFAULT_RAMP_SCHEDULE

    text = normalize_text(text)

    results = {
        "ttft_ms": [],
        "first_chunk_duration_ms": [],
        "tokens_in_first_chunk": ramp_schedule[0][0],
        "cfm_steps_first_chunk": ramp_schedule[0][2],
    }

    for run in range(num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        start = time.perf_counter()

        for chunk, metrics in model.generate_stream(
            text,
            temperature=0.3,
            ramp_schedule=ramp_schedule,
        ):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            ttft = (time.perf_counter() - start) * 1000

            chunk_duration = chunk.shape[-1] / 24000 * 1000  # ms

            results["ttft_ms"].append(ttft)
            results["first_chunk_duration_ms"].append(chunk_duration)

            print(f"  Run {run+1}: TTFT={ttft:.1f}ms, chunk={chunk_duration:.1f}ms audio")
            break  # Only measure first chunk

    # Compute stats
    results["ttft_mean_ms"] = np.mean(results["ttft_ms"])
    results["ttft_std_ms"] = np.std(results["ttft_ms"])
    results["ttft_min_ms"] = np.min(results["ttft_ms"])
    results["ttft_max_ms"] = np.max(results["ttft_ms"])
    results["chunk_duration_mean_ms"] = np.mean(results["first_chunk_duration_ms"])

    return results


def print_results(results: dict, mode: str):
    """Print formatted results."""
    print(f"\n{'='*60}")
    print(f"TTFT Benchmark Results - {mode.upper()} mode")
    print(f"{'='*60}")
    print(f"First chunk config:")
    print(f"  Tokens:     {results['tokens_in_first_chunk']}")
    print(f"  CFM steps:  {results['cfm_steps_first_chunk']}")
    print(f"\nTime to First Token (TTFT):")
    print(f"  Mean:  {results['ttft_mean_ms']:.1f} ms")
    print(f"  Std:   {results['ttft_std_ms']:.1f} ms")
    print(f"  Min:   {results['ttft_min_ms']:.1f} ms")
    print(f"  Max:   {results['ttft_max_ms']:.1f} ms")
    print(f"\nFirst chunk audio duration:")
    print(f"  Mean:  {results['chunk_duration_mean_ms']:.1f} ms")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="TTFT Benchmark")
    parser.add_argument(
        "--mode",
        choices=["gpu", "cpu", "hybrid"],
        default="gpu",
        help="Device mode: gpu (all GPU), cpu (all CPU), hybrid (T3=CPU, S3Gen=GPU)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of benchmark runs",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="voice_ref.wav",
        help="Voice reference file",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, this is a test of the time to first token latency.",
        help="Text to synthesize",
    )
    args = parser.parse_args()

    # Check voice file exists
    if not Path(args.voice).exists():
        print(f"Error: Voice file not found: {args.voice}")
        return

    # Load model based on mode
    if args.mode == "gpu":
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            args.mode = "cpu"
            model = load_model_standard("cpu")
        else:
            model = load_model_standard("cuda")
    elif args.mode == "cpu":
        model = load_model_standard("cpu")
    elif args.mode == "hybrid":
        if not torch.cuda.is_available():
            print("CUDA not available for hybrid mode, falling back to CPU")
            args.mode = "cpu"
            model = load_model_standard("cpu")
        else:
            model = load_model_hybrid()

    # Warmup
    warmup(model, args.voice)

    # Run benchmark with default schedule
    print(f"\n--- Benchmark with DEFAULT schedule ---")
    print(f"Text: '{args.text}'")
    print(f"Runs: {args.runs}")

    results = benchmark_ttft(
        model,
        args.text,
        num_runs=args.runs,
        ramp_schedule=DEFAULT_RAMP_SCHEDULE,
    )
    print_results(results, args.mode)

    # Also test with minimal first chunk for comparison
    print(f"\n--- Benchmark with MINIMAL first chunk (2 tokens, 1 CFM) ---")
    minimal_schedule = [
        (2, 0, 1),  # Ultra minimal first chunk
        (8, 2, 3),
        (16, 10, 5),
        (32, 26, 7),
    ]

    results_minimal = benchmark_ttft(
        model,
        args.text,
        num_runs=args.runs,
        ramp_schedule=minimal_schedule,
    )
    print_results(results_minimal, f"{args.mode} (minimal)")

    # Summary comparison
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"Default schedule (4 tokens, 1 CFM):  {results['ttft_mean_ms']:.1f} ms")
    print(f"Minimal schedule (2 tokens, 1 CFM):  {results_minimal['ttft_mean_ms']:.1f} ms")
    print(f"Difference: {results['ttft_mean_ms'] - results_minimal['ttft_mean_ms']:.1f} ms")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
