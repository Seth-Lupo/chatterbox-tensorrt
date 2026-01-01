"""
Benchmark: Time to First Token (TTFT)

Measures latency to first audio chunk.

Usage:
    python benchmark_ttft.py
    python benchmark_ttft.py --runs 10
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


def benchmark_ttft(
    model: ChatterboxTurboTTS,
    text: str,
    num_runs: int = 5,
    ramp_schedule=None,
) -> dict:
    """Benchmark time to first chunk."""
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

    results["ttft_mean_ms"] = np.mean(results["ttft_ms"])
    results["ttft_std_ms"] = np.std(results["ttft_ms"])
    results["ttft_min_ms"] = np.min(results["ttft_ms"])
    results["ttft_max_ms"] = np.max(results["ttft_ms"])
    results["chunk_duration_mean_ms"] = np.mean(results["first_chunk_duration_ms"])

    return results


def print_results(results: dict, label: str):
    """Print formatted results."""
    print(f"\n{'='*60}")
    print(f"TTFT Benchmark Results - {label}")
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
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs")
    parser.add_argument("--voice", type=str, default="voice_ref.wav", help="Voice reference file")
    parser.add_argument("--text", type=str, default="Hello, this is a test of the time to first token latency.", help="Text to synthesize")
    args = parser.parse_args()

    if not Path(args.voice).exists():
        print(f"Error: Voice file not found: {args.voice}")
        return

    # Load model on CPU, then move S3Gen to GPU
    print("Loading model (T3 on CPU)...")
    model = ChatterboxTurboTTS.from_pretrained(device="cpu")

    # Prepare conditionals while still on CPU
    print("Preparing conditionals...")
    model.prepare_conditionals(args.voice, exaggeration=0.5)

    print("Moving S3Gen to GPU...")
    model.s3gen = model.s3gen.to("cuda")

    # Move S3Gen conditioning to GPU
    for k, v in model.conds.gen.items():
        if torch.is_tensor(v):
            model.conds.gen[k] = v.to("cuda")

    print("Compiling S3Gen...")
    model.s3gen.flow = torch.compile(model.s3gen.flow)
    model._compiled = True

    # Warmup
    print("Warming up...")
    for _ in model.generate_stream("Hello.", ramp_schedule=[(4, 0, 1)]):
        break
    print("Warmup complete")

    # Run benchmark
    print(f"\n--- Benchmark with DEFAULT schedule ---")
    print(f"Text: '{args.text}'")
    print(f"Runs: {args.runs}")

    results = benchmark_ttft(model, args.text, num_runs=args.runs)
    print_results(results, "T3=CPU, S3Gen=GPU")

    # Minimal schedule comparison
    print(f"\n--- Benchmark with MINIMAL first chunk (2 tokens, 1 CFM) ---")
    minimal_schedule = [
        (2, 0, 1),
        (8, 2, 3),
        (16, 10, 5),
        (32, 26, 7),
    ]

    results_minimal = benchmark_ttft(model, args.text, num_runs=args.runs, ramp_schedule=minimal_schedule)
    print_results(results_minimal, "T3=CPU, S3Gen=GPU (minimal)")

    # Summary
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"Default (4 tokens, 1 CFM):  {results['ttft_mean_ms']:.1f} ms")
    print(f"Minimal (2 tokens, 1 CFM):  {results_minimal['ttft_mean_ms']:.1f} ms")
    print(f"Difference: {results['ttft_mean_ms'] - results_minimal['ttft_mean_ms']:.1f} ms")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
