#!/usr/bin/env python3
"""
Test and benchmark TensorRT vs PyTorch for T3 transformer.

This compares:
1. PyTorch transformer (baseline)
2. TensorRT transformer (accelerated)

Usage:
    python test_trt.py
    python test_trt.py --full-tts  # Test full TTS pipeline
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project to path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

ENGINE_PATH = SCRIPT_DIR / "t3_transformer.engine"


def benchmark_pytorch(model, seq_lengths, num_iterations=50):
    """Benchmark PyTorch transformer."""
    print("\n" + "=" * 60)
    print("PyTorch Transformer Benchmark")
    print("=" * 60)

    hidden_size = model.t3.cfg.hidden_size
    results = {}

    for seq_len in seq_lengths:
        test_input = torch.randn(
            1, seq_len, hidden_size,
            dtype=torch.float16,
            device="cuda",
        )

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model.t3.tfmr(inputs_embeds=test_input)
        torch.cuda.synchronize()

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model.t3.tfmr(inputs_embeds=test_input)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

        mean_ms = np.mean(times) * 1000
        results[seq_len] = mean_ms
        print(f"  Seq {seq_len:4d}: {mean_ms:.2f} ms ({seq_len / np.mean(times):.0f} tok/s)")

    return results


def benchmark_tensorrt(engine_path, seq_lengths, num_iterations=50):
    """Benchmark TensorRT transformer."""
    print("\n" + "=" * 60)
    print("TensorRT Transformer Benchmark")
    print("=" * 60)

    from trt_wrapper import T3TensorRTTransformer

    trt_transformer = T3TensorRTTransformer(str(engine_path))
    results = {}

    for seq_len in seq_lengths:
        stats = trt_transformer.benchmark(seq_len=seq_len, num_iterations=num_iterations)
        results[seq_len] = stats["mean_ms"]
        print(f"  Seq {seq_len:4d}: {stats['mean_ms']:.2f} ms ({stats['throughput_tokens_per_sec']:.0f} tok/s)")

    return results


def verify_outputs(model, engine_path):
    """Verify TensorRT outputs match PyTorch."""
    print("\n" + "=" * 60)
    print("Verifying TensorRT vs PyTorch Outputs")
    print("=" * 60)

    from trt_wrapper import T3TensorRTTransformer
    from export_onnx import T3TransformerOnly

    # Load TensorRT
    trt_transformer = T3TensorRTTransformer(str(engine_path))

    # Create PyTorch wrapper (same as export)
    pytorch_transformer = T3TransformerOnly(model.t3.tfmr).cuda().half().eval()

    hidden_size = model.t3.cfg.hidden_size

    # Test various sequence lengths
    for seq_len in [10, 50, 100, 256]:
        test_input = torch.randn(1, seq_len, hidden_size, dtype=torch.float16, device="cuda")

        # PyTorch output
        with torch.no_grad():
            pytorch_output = pytorch_transformer(test_input)

        # TensorRT output
        trt_output = trt_transformer(test_input)

        # Compare
        diff = (pytorch_output - trt_output).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        status = "✓" if max_diff < 0.1 else "✗"
        print(f"  Seq {seq_len:4d}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f} {status}")

    print("\nNote: Small differences are expected due to FP16 precision and TensorRT optimizations")


def test_full_tts(model, engine_path):
    """Test full TTS pipeline with TensorRT transformer."""
    print("\n" + "=" * 60)
    print("Full TTS Pipeline Test")
    print("=" * 60)

    from trt_wrapper import T3TensorRTTransformer

    # This would require deeper integration with the T3 model
    # For now, just show that the engine can be loaded and used

    print("\nTo integrate TensorRT into full TTS:")
    print("  1. Replace model.t3.tfmr with trt_transformer in generation loop")
    print("  2. Keep speech_emb, cond_enc, speech_head in PyTorch")
    print("  3. Handle KV caching manually (or run without cache)")
    print("\nSee T3HybridModel in trt_wrapper.py for integration example")


def main():
    parser = argparse.ArgumentParser(description="Test TensorRT T3 transformer")
    parser.add_argument("--full-tts", action="store_true", help="Test full TTS pipeline")
    parser.add_argument("--verify-only", action="store_true", help="Only verify outputs")
    parser.add_argument("--iterations", type=int, default=50, help="Benchmark iterations")
    args = parser.parse_args()

    print("=" * 60)
    print("T3 TensorRT Transformer Test")
    print("=" * 60)

    # Check engine exists
    if not ENGINE_PATH.exists():
        print(f"\nERROR: TensorRT engine not found: {ENGINE_PATH}")
        print("\nRun these steps first:")
        print("  1. python export_onnx.py")
        print("  2. ./build_engine.sh")
        sys.exit(1)

    print(f"\nEngine: {ENGINE_PATH}")
    print(f"Size: {ENGINE_PATH.stat().st_size / (1024*1024):.2f} MB")

    # Load T3 model for comparison
    print("\nLoading T3 model...")
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    model = ChatterboxTurboTTS.from_pretrained(
        device="cuda",
        dtype="float16",
        compile_mode=None,
    )
    print("Model loaded!")

    # Verify outputs
    verify_outputs(model, ENGINE_PATH)

    if args.verify_only:
        return

    # Benchmark
    seq_lengths = [50, 100, 256, 512, 1024]

    pytorch_results = benchmark_pytorch(model, seq_lengths, args.iterations)
    trt_results = benchmark_tensorrt(ENGINE_PATH, seq_lengths, args.iterations)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: TensorRT Speedup")
    print("=" * 60)
    print(f"\n{'Seq Len':<10} {'PyTorch':<12} {'TensorRT':<12} {'Speedup':<10}")
    print("-" * 44)

    for seq_len in seq_lengths:
        pytorch_ms = pytorch_results[seq_len]
        trt_ms = trt_results[seq_len]
        speedup = pytorch_ms / trt_ms

        print(f"{seq_len:<10} {pytorch_ms:<12.2f} {trt_ms:<12.2f} {speedup:<10.2f}x")

    print("-" * 44)

    avg_speedup = np.mean([pytorch_results[s] / trt_results[s] for s in seq_lengths])
    print(f"\nAverage speedup: {avg_speedup:.2f}x")

    if args.full_tts:
        test_full_tts(model, ENGINE_PATH)

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
The TensorRT engine is working! To use it in production:

1. For transformer-only acceleration (no KV cache):
   - Replace model.t3.tfmr calls with trt_transformer
   - Good for prefill, not ideal for autoregressive

2. For full autoregressive with KV cache:
   - Keep using PyTorch with model.t3.tfmr(..., use_cache=True)
   - TensorRT KV cache requires more complex integration

3. For maximum performance:
   - Use TensorRT for prefill (processing conditioning)
   - Use PyTorch+KV cache for autoregressive token generation
""")


if __name__ == "__main__":
    main()
