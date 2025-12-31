#!/usr/bin/env python3
"""
Generate audio using T3 TTS.

NOTE: The TensorRT engine accelerates the transformer forward pass by ~2.3x,
but T3's autoregressive generation uses KV caching. Our TensorRT engine
is "prefill-only" (no KV cache), so we use PyTorch for actual generation.

For full TensorRT acceleration with KV cache, you'd need TensorRT-LLM.

Usage:
    python generate_audio.py "Hello, this is a test."
    python generate_audio.py "Hello world" --output hello.wav
    python generate_audio.py "Hello world" --voice voice.wav
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import numpy as np
from scipy.io import wavfile

# Add project to path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def generate_audio(
    text: str,
    output_path: Path,
    audio_prompt_path: str = None,
):
    """Generate audio and save to file."""
    print("=" * 60)
    print("T3 Audio Generation")
    print("=" * 60)

    # Load model
    print("\nLoading T3 model...")
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    model = ChatterboxTurboTTS.from_pretrained(
        device="cuda",
        dtype="float16",
        compile_mode=None,  # No torch.compile for simplicity
    )
    print("Model loaded!")

    # Generate
    print(f"\nGenerating audio for: \"{text}\"")
    if audio_prompt_path:
        print(f"  Voice reference: {audio_prompt_path}")

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # Collect audio chunks from streaming generation
    audio_chunks = []
    for chunk, metrics in model.generate_stream(
        text=text,
        audio_prompt_path=audio_prompt_path,
    ):
        audio_chunks.append(chunk)

    # Concatenate all chunks
    audio = torch.cat(audio_chunks, dim=-1)
    sr = model.sr

    torch.cuda.synchronize()
    gen_time = time.perf_counter() - start_time

    # Calculate stats
    audio_duration = audio.shape[-1] / sr
    rtf = gen_time / audio_duration

    print(f"\nGeneration complete!")
    print(f"  Audio duration: {audio_duration:.2f}s")
    print(f"  Generation time: {gen_time:.2f}s")
    print(f"  Real-time factor: {rtf:.2f}x (< 1.0 = faster than real-time)")

    # Save audio
    print(f"\nSaving to: {output_path}")

    # Ensure audio is on CPU and correct shape
    audio_cpu = audio.cpu()
    if audio_cpu.dim() == 2:
        audio_cpu = audio_cpu.squeeze(0)  # Remove batch dim for mono

    # Convert to int16 for WAV
    audio_np = audio_cpu.numpy()
    audio_np = np.clip(audio_np, -1.0, 1.0)  # Clip to valid range
    audio_int16 = (audio_np * 32767).astype(np.int16)

    wavfile.write(str(output_path), sr, audio_int16)
    print(f"  Saved! ({output_path.stat().st_size / 1024:.1f} KB)")

    return audio, sr, gen_time


def demo_tensorrt_transformer():
    """Demonstrate TensorRT transformer on standalone embeddings."""
    print("\n" + "=" * 60)
    print("TensorRT Transformer Demo")
    print("=" * 60)

    engine_path = SCRIPT_DIR / "t3_transformer.engine"
    if not engine_path.exists():
        print(f"TensorRT engine not found: {engine_path}")
        print("Run ./build_engine.sh first")
        return

    from trt_wrapper import T3TensorRTTransformer

    # Load TensorRT engine
    trt = T3TensorRTTransformer(str(engine_path))

    # Test with random embeddings
    print("\nRunning TensorRT inference on random embeddings...")
    test_input = torch.randn(1, 256, 1024, dtype=torch.float16, device="cuda")

    torch.cuda.synchronize()
    start = time.perf_counter()

    output = trt(test_input)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Time: {elapsed*1000:.2f} ms")
    print(f"  Throughput: {256/elapsed:.0f} tokens/sec")


def main():
    parser = argparse.ArgumentParser(description="Generate audio with T3 TTS")
    parser.add_argument("text", type=str, nargs="?",
                        default="Hello! This is a test of the text to speech system.",
                        help="Text to synthesize")
    parser.add_argument("--output", "-o", type=Path, default=SCRIPT_DIR / "output.wav",
                        help="Output WAV file path")
    parser.add_argument("--voice", type=str, default=None,
                        help="Path to voice reference audio for cloning")
    parser.add_argument("--demo-trt", action="store_true",
                        help="Demo TensorRT transformer (doesn't generate audio)")
    args = parser.parse_args()

    if args.demo_trt:
        demo_tensorrt_transformer()
    else:
        generate_audio(args.text, args.output, args.voice)

    print("\n" + "=" * 60)
    print("NOTES ON TENSORRT INTEGRATION")
    print("=" * 60)
    print("""
The TensorRT engine we built provides 2.3x speedup for the transformer
forward pass. However, T3 uses autoregressive generation with KV caching:

  - Each token generation only processes 1 new token
  - KV cache stores previous keys/values to avoid recomputation
  - Our TensorRT engine is "prefill-only" (processes full sequence)

For full TensorRT acceleration of autoregressive generation, you'd need:
  1. TensorRT-LLM with native KV cache support, OR
  2. Modify engine to accept KV cache tensors as inputs/outputs

The PyTorch KV cache implementation is already fast (~0.3-0.5 RTF).
TensorRT would mainly help for batch processing or very long sequences.
""")


if __name__ == "__main__":
    main()
