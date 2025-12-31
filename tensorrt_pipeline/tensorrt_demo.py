#!/usr/bin/env python3
"""
TensorRT-Optimized Chatterbox Turbo Demo

This script demonstrates streaming TTS with TensorRT optimization.

Usage:
    # Basic usage
    python tensorrt_demo.py --text "Hello world"

    # With voice cloning
    python tensorrt_demo.py --audio_prompt reference.wav --text "Clone this voice"

    # Custom engine directory
    python tensorrt_demo.py --engines_dir /path/to/engines --text "Hello"
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torchaudio as ta

# Add pipeline directory to path
sys.path.insert(0, str(Path(__file__).parent))

from tensorrt_inference import ChatterboxTurboTRT, StreamingMetrics


def print_banner():
    """Print startup banner."""
    print()
    print("=" * 60)
    print("  Chatterbox Turbo - TensorRT Optimized")
    print("=" * 60)


def print_gpu_info():
    """Print GPU information."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")

        # Check compute capability
        cc = torch.cuda.get_device_capability(0)
        print(f"  Compute Capability: {cc[0]}.{cc[1]}")

        # TensorRT works best on cc 7.0+
        if cc[0] < 7:
            print("  Warning: Compute capability < 7.0, TensorRT may not be optimal")
    else:
        print("  Warning: CUDA not available, running on CPU")


def main():
    parser = argparse.ArgumentParser(
        description="TensorRT-Optimized Chatterbox Turbo TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tensorrt_demo.py --text "Hello, this is TensorRT accelerated speech!"
  python tensorrt_demo.py --audio_prompt voice.wav --text "Clone this voice"
  python tensorrt_demo.py --dtype float16 --chunk_size 40 --text "Fast generation"
        """
    )

    # Input/Output
    parser.add_argument("--text", type=str, required=True,
                        help="Text to synthesize")
    parser.add_argument("--audio_prompt", type=str, default=None,
                        help="Reference audio for voice cloning (>5 seconds)")
    parser.add_argument("--output", type=str, default="output_trt.wav",
                        help="Output audio file")

    # Model options
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "float32"],
                        help="Data type for inference")
    parser.add_argument("--engines_dir", type=str, default=None,
                        help="Directory containing TensorRT engines")

    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=1000,
                        help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p (nucleus) sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                        help="Repetition penalty")

    # Streaming parameters
    parser.add_argument("--chunk_size", type=int, default=25,
                        help="Tokens per audio chunk")
    parser.add_argument("--context_window", type=int, default=50,
                        help="Context window for audio coherence")

    # Misc
    parser.add_argument("--benchmark", action="store_true",
                        help="Run multiple iterations for benchmarking")
    parser.add_argument("--benchmark_iterations", type=int, default=5,
                        help="Number of benchmark iterations")
    parser.add_argument("--no_save", action="store_true",
                        help="Don't save audio output")

    args = parser.parse_args()

    print_banner()
    print_gpu_info()

    # Load model
    print("\nLoading model...")
    load_start = time.time()

    model = ChatterboxTurboTRT.from_pretrained(
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=args.dtype,
        engines_dir=args.engines_dir,
    )

    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s")

    # Run generation
    if args.benchmark:
        run_benchmark(model, args)
    else:
        run_single(model, args)


def run_single(model, args):
    """Run single generation."""
    print(f"\nGenerating speech for: '{args.text}'")
    print("-" * 60)

    audio_chunks = []
    final_metrics = None

    for audio_chunk, metrics in model.generate_stream(
        text=args.text,
        audio_prompt_path=args.audio_prompt,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        chunk_size=args.chunk_size,
        context_window=args.context_window,
    ):
        audio_chunks.append(audio_chunk)
        final_metrics = metrics

        # Print progress
        if metrics.latency_to_first_chunk and metrics.chunk_count == 1:
            print(f"  First chunk latency: {metrics.latency_to_first_chunk:.3f}s")

        chunk_duration = audio_chunk.shape[-1] / model.sr
        print(f"  Chunk {metrics.chunk_count}: {chunk_duration:.2f}s of audio")

    # Combine chunks
    if audio_chunks:
        final_audio = torch.cat(audio_chunks, dim=-1)

        # Print final metrics
        print("-" * 60)
        print("Results:")
        if final_metrics.latency_to_first_chunk:
            print(f"  First chunk latency: {final_metrics.latency_to_first_chunk:.3f}s")
        if final_metrics.total_generation_time:
            print(f"  Total generation time: {final_metrics.total_generation_time:.3f}s")
        if final_metrics.total_audio_duration:
            print(f"  Total audio duration: {final_metrics.total_audio_duration:.3f}s")
        if final_metrics.rtf:
            rtf_status = "REAL-TIME" if final_metrics.rtf < 1.0 else "slower than real-time"
            print(f"  Real-time factor: {final_metrics.rtf:.3f} ({rtf_status})")
        print(f"  Chunks generated: {final_metrics.chunk_count}")

        # Save audio
        if not args.no_save:
            ta.save(args.output, final_audio, model.sr)
            print(f"\nSaved to: {args.output}")
    else:
        print("No audio generated!")


def run_benchmark(model, args):
    """Run benchmark across multiple iterations."""
    print(f"\nRunning benchmark ({args.benchmark_iterations} iterations)...")
    print(f"Text: '{args.text[:50]}...'")
    print("-" * 60)

    latencies = []
    rtfs = []
    gen_times = []

    for i in range(args.benchmark_iterations):
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        audio_chunks = []
        for audio_chunk, metrics in model.generate_stream(
            text=args.text,
            audio_prompt_path=args.audio_prompt,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            chunk_size=args.chunk_size,
            context_window=args.context_window,
        ):
            audio_chunks.append(audio_chunk)

        if metrics.latency_to_first_chunk:
            latencies.append(metrics.latency_to_first_chunk)
        if metrics.rtf:
            rtfs.append(metrics.rtf)
        if metrics.total_generation_time:
            gen_times.append(metrics.total_generation_time)

        print(f"  Iteration {i+1}: latency={metrics.latency_to_first_chunk:.3f}s, RTF={metrics.rtf:.3f}")

    # Print summary
    print("\n" + "=" * 60)
    print("Benchmark Results (excluding warmup)")
    print("=" * 60)

    if len(latencies) > 1:
        # Exclude first iteration (warmup)
        latencies = latencies[1:]
        rtfs = rtfs[1:]
        gen_times = gen_times[1:]

    import statistics

    if latencies:
        print(f"  First chunk latency:")
        print(f"    Mean: {statistics.mean(latencies):.3f}s")
        print(f"    Min:  {min(latencies):.3f}s")
        print(f"    Max:  {max(latencies):.3f}s")
        if len(latencies) > 1:
            print(f"    Std:  {statistics.stdev(latencies):.3f}s")

    if rtfs:
        print(f"\n  Real-time factor (RTF):")
        print(f"    Mean: {statistics.mean(rtfs):.3f}")
        print(f"    Min:  {min(rtfs):.3f}")
        print(f"    Max:  {max(rtfs):.3f}")

    if gen_times:
        print(f"\n  Total generation time:")
        print(f"    Mean: {statistics.mean(gen_times):.3f}s")

    # Performance rating
    if rtfs:
        avg_rtf = statistics.mean(rtfs)
        if avg_rtf < 0.3:
            rating = "EXCELLENT"
        elif avg_rtf < 0.5:
            rating = "VERY GOOD"
        elif avg_rtf < 1.0:
            rating = "GOOD (real-time capable)"
        else:
            rating = "NEEDS OPTIMIZATION"
        print(f"\n  Performance rating: {rating}")


if __name__ == "__main__":
    main()
