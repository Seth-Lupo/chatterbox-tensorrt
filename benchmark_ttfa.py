#!/usr/bin/env python3
"""
Time to First Audio (TTFA) Benchmark for Chatterbox Turbo Streaming TTS

Measures latency from request initiation to first audio chunk delivery.
"""

import argparse
import gc
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))
from chatterbox import ChatterboxTurboTTS


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""
    ttfa_mean_ms: float
    ttfa_std_ms: float
    ttfa_min_ms: float
    ttfa_max_ms: float
    ttfa_p50_ms: float
    ttfa_p95_ms: float
    ttfa_p99_ms: float
    total_latency_mean_ms: float
    rtf_mean: float
    samples: int


def percentile(data: list[float], p: float) -> float:
    """Calculate percentile."""
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def run_single_benchmark(
    model: ChatterboxTurboTTS,
    text: str,
    chunk_size: int = 25,
) -> tuple[float, float, float]:
    """
    Run a single streaming generation and measure TTFA.

    Returns: (ttfa_ms, total_latency_ms, rtf)
    """
    start = time.perf_counter()
    ttfa = None
    total_latency = None
    rtf = None

    for audio_chunk, metrics in model.generate_stream(
        text=text,
        chunk_size=chunk_size,
    ):
        if ttfa is None:
            ttfa = (time.perf_counter() - start) * 1000
        total_latency = metrics.total_generation_time
        rtf = metrics.rtf

    if total_latency is None:
        total_latency = (time.perf_counter() - start)

    return ttfa, (total_latency or 0) * 1000, rtf or 0


def run_benchmark(
    model: ChatterboxTurboTTS,
    text: str,
    warmup_runs: int = 3,
    benchmark_runs: int = 10,
    chunk_size: int = 25,
) -> BenchmarkResult:
    """
    Run full benchmark with warmup and multiple iterations.
    """
    # Warmup
    for _ in range(warmup_runs):
        run_single_benchmark(model, text, chunk_size)
        torch.cuda.synchronize() if torch.cuda.is_available() else None

    # Benchmark
    ttfa_samples = []
    total_latency_samples = []
    rtf_samples = []

    for _ in range(benchmark_runs):
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        ttfa, total_latency, rtf = run_single_benchmark(model, text, chunk_size)
        ttfa_samples.append(ttfa)
        total_latency_samples.append(total_latency)
        rtf_samples.append(rtf)

        torch.cuda.synchronize() if torch.cuda.is_available() else None

    return BenchmarkResult(
        ttfa_mean_ms=statistics.mean(ttfa_samples),
        ttfa_std_ms=statistics.stdev(ttfa_samples) if len(ttfa_samples) > 1 else 0,
        ttfa_min_ms=min(ttfa_samples),
        ttfa_max_ms=max(ttfa_samples),
        ttfa_p50_ms=percentile(ttfa_samples, 50),
        ttfa_p95_ms=percentile(ttfa_samples, 95),
        ttfa_p99_ms=percentile(ttfa_samples, 99),
        total_latency_mean_ms=statistics.mean(total_latency_samples),
        rtf_mean=statistics.mean(rtf_samples),
        samples=benchmark_runs,
    )


def print_results(result: BenchmarkResult, text: str, label: str = ""):
    """Print formatted benchmark results."""
    print(f"\n{'=' * 60}")
    if label:
        print(f"  {label}")
    print(f"  Text: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
    print(f"  Characters: {len(text)}")
    print(f"{'=' * 60}")
    print(f"  Time to First Audio (TTFA):")
    print(f"    Mean:   {result.ttfa_mean_ms:>8.2f} ms")
    print(f"    Std:    {result.ttfa_std_ms:>8.2f} ms")
    print(f"    Min:    {result.ttfa_min_ms:>8.2f} ms")
    print(f"    Max:    {result.ttfa_max_ms:>8.2f} ms")
    print(f"    P50:    {result.ttfa_p50_ms:>8.2f} ms")
    print(f"    P95:    {result.ttfa_p95_ms:>8.2f} ms")
    print(f"    P99:    {result.ttfa_p99_ms:>8.2f} ms")
    print(f"  Total Generation:")
    print(f"    Latency:{result.total_latency_mean_ms:>8.2f} ms")
    print(f"    RTF:    {result.rtf_mean:>8.3f}x")
    print(f"  Samples:  {result.samples}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Time to First Audio (TTFA) for streaming TTS"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda/cpu/mps)"
    )
    parser.add_argument(
        "--dtype", type=str, default="float16", choices=["float32", "float16", "bfloat16"],
        help="Model precision"
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="Enable torch.compile for S3Gen"
    )
    parser.add_argument(
        "--warmup", type=int, default=3,
        help="Number of warmup runs"
    )
    parser.add_argument(
        "--runs", type=int, default=10,
        help="Number of benchmark runs per test"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=25,
        help="Tokens per audio chunk"
    )
    parser.add_argument(
        "--audio-prompt", type=str, default=None,
        help="Path to voice reference audio (optional)"
    )
    parser.add_argument(
        "--text", type=str, default=None,
        help="Custom text to benchmark (overrides default tests)"
    )
    args = parser.parse_args()

    # Load model
    print(f"\nLoading model on {args.device} with {args.dtype}...")
    model = ChatterboxTurboTTS.from_pretrained(
        device=args.device,
        dtype=args.dtype,
        compile_mode="default" if args.compile else None,
    )

    # Prepare voice conditionals if provided
    if args.audio_prompt:
        print(f"Loading voice reference: {args.audio_prompt}")
        model.prepare_conditionals(args.audio_prompt)

    print(f"Model loaded. Compile: {args.compile}")
    print(f"Warmup runs: {args.warmup}, Benchmark runs: {args.runs}")
    print(f"Chunk size: {args.chunk_size} tokens")

    # Test cases
    if args.text:
        test_cases = [("Custom", args.text)]
    else:
        test_cases = [
            ("Short", "Hello, how are you today?"),
            ("Medium", "The quick brown fox jumps over the lazy dog. This is a test of the streaming text to speech system."),
            ("Long", "In the beginning, the universe was created. This has made a lot of people very angry and been widely regarded as a bad move. Many races believe that it was created by some sort of god, though the Jatravartid people of Viltvodle Six believe that the entire universe was sneezed out of the nose of a being called the Great Green Arkleseizure."),
        ]

    # Run benchmarks
    print("\n" + "=" * 60)
    print("  TTFA BENCHMARK RESULTS")
    print("=" * 60)

    for label, text in test_cases:
        result = run_benchmark(
            model=model,
            text=text,
            warmup_runs=args.warmup,
            benchmark_runs=args.runs,
            chunk_size=args.chunk_size,
        )
        print_results(result, text, label)

    # Summary
    print("\n" + "-" * 60)
    print("  Configuration Summary")
    print("-" * 60)
    print(f"  Device:     {args.device}")
    print(f"  Dtype:      {args.dtype}")
    print(f"  Compiled:   {args.compile}")
    print(f"  Chunk size: {args.chunk_size}")
    if torch.cuda.is_available():
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
