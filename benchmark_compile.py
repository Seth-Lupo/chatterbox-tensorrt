#!/usr/bin/env python3
"""
Benchmark: Chatterbox Turbo with torch.compile

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

import numpy as np
import torch
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parent / "src"))
from chatterbox.tts_turbo import ChatterboxTurboTTS

TEST_TEXTS = [
    "Hello, this is a test of the text to speech system.",
    "The quick brown fox jumps over the lazy dog.",
    "Welcome to the future of artificial intelligence.",
    "Today we are testing streaming audio generation.",
    "This benchmark measures latency to first audio chunk.",
]


def measure_first_chunk_latency(model, text: str, audio_prompt: str = None) -> float:
    """Measure time to first audio chunk."""
    torch.cuda.synchronize()
    start = time.perf_counter()

    gen = model.generate_stream(text=text, audio_prompt_path=audio_prompt)
    next(gen)
    torch.cuda.synchronize()
    latency = time.perf_counter() - start

    for _ in gen:
        pass

    return latency


def main():
    parser = argparse.ArgumentParser(description="Benchmark Chatterbox Turbo")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--audio_prompt", type=str, default=None)
    parser.add_argument("--output", type=str, default="output.wav")
    args = parser.parse_args()

    print("=" * 60)
    print("Chatterbox Turbo Benchmark")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("ERROR: CUDA not available")
        sys.exit(1)

    # Load and compile model
    print("\nLoading model...")
    load_start = time.perf_counter()
    model = ChatterboxTurboTTS.from_pretrained(device="cuda")
    model.compile()
    print(f"Loaded in {time.perf_counter() - load_start:.2f}s")

    # Warmup
    print("\nWarmup...")
    for i in range(3):
        measure_first_chunk_latency(model, TEST_TEXTS[i], args.audio_prompt)
    print("Warmup complete.")

    # Benchmark
    print(f"\nRunning {args.iterations} iterations...")
    latencies = []
    for i in range(args.iterations):
        text = TEST_TEXTS[i % len(TEST_TEXTS)]
        latency = measure_first_chunk_latency(model, text, args.audio_prompt)
        latencies.append(latency)
        print(f"  {i+1}/{args.iterations}: {latency:.3f}s")
        torch.cuda.empty_cache()

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Mean:   {statistics.mean(latencies):.3f}s")
    print(f"Min:    {min(latencies):.3f}s")
    print(f"Max:    {max(latencies):.3f}s")
    if len(latencies) > 1:
        print(f"Stdev:  {statistics.stdev(latencies):.3f}s")

    mean = statistics.mean(latencies)
    if mean < 0.3:
        print("\nPerformance: EXCELLENT (< 300ms)")
    elif mean < 0.5:
        print("\nPerformance: VERY GOOD (< 500ms)")
    elif mean < 1.0:
        print("\nPerformance: GOOD (< 1s)")
    else:
        print("\nPerformance: NEEDS WORK (> 1s)")

    # Generate sample
    print(f"\nGenerating sample to {args.output}...")
    sample_text = "You got to see this! I achieved sub one hundred millisecond latency using exponential chunking! [laugh]"

    chunks = []
    for chunk, metrics in model.generate_stream(text=sample_text, audio_prompt_path=args.audio_prompt):
        chunks.append(chunk)
        status = "AHEAD" if metrics.buffer_ahead > 0 else "BEHIND"
        print(f"  Chunk {metrics.chunk_count}: buffer={metrics.buffer_ahead:+.2f}s [{status}]")

    audio = torch.cat(chunks, dim=-1).squeeze().cpu().numpy()
    wavfile.write(args.output, model.sr, (audio * 32767).astype(np.int16))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
