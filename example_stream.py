"""
Chatterbox Turbo Streaming TTS Example

Basic usage:
    python example_stream.py

With optimizations (recommended for GPU):
    python example_stream.py --dtype float16 --compile reduce-overhead

With voice cloning:
    python example_stream.py --audio_prompt path/to/reference.wav

Maximum performance (G6/L4 GPU):
    python example_stream.py --dtype float16 --compile max-autotune --chunk_size 40
"""
import sys
sys.path.insert(0, "/Users/sethlupo/Public/new-bot/martha/src")

import argparse
import numpy as np
import torch
from scipy.io import wavfile
from chatterbox import ChatterboxTurboTTS


def main():
    parser = argparse.ArgumentParser(description="Chatterbox Turbo Streaming TTS")
    parser.add_argument("--text", type=str, default="Hello! This is a streaming text to speech demonstration using Chatterbox Turbo.")
    parser.add_argument("--audio_prompt", type=str, default=None, help="Path to reference audio for voice cloning (must be >5 seconds)")
    parser.add_argument("--output", type=str, default="output.wav", help="Output audio file path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Optimization options
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"],
                        help="Data type: float32 (accurate), float16 (fast, slight quality loss), bfloat16 (balanced)")
    parser.add_argument("--compile", type=str, default=None,
                        choices=["default", "reduce-overhead", "max-autotune", "tensorrt"],
                        help="Compilation mode for faster inference")

    # Streaming parameters
    parser.add_argument("--chunk_size", type=int, default=25,
                        help="Tokens per audio chunk (higher = fewer S3Gen calls, more latency)")
    parser.add_argument("--context_window", type=int, default=50,
                        help="Context tokens for audio coherence (higher = smoother, slower)")

    args = parser.parse_args()

    print(f"Loading model on {args.device}...")
    print(f"  dtype: {args.dtype}")
    print(f"  compile: {args.compile or 'none'}")

    model = ChatterboxTurboTTS.from_pretrained(
        device=args.device,
        dtype=args.dtype,
        compile_mode=args.compile,
    )

    # Warmup run (important for compiled models)
    if args.compile:
        print("Warming up compiled model...")
        warmup_text = "Hello."
        for _ in model.generate_stream(warmup_text, chunk_size=args.chunk_size):
            pass
        print("Warmup complete.")

    print(f"\nGenerating speech for: '{args.text}'")
    print("-" * 50)

    audio_chunks = []
    for audio_chunk, metrics in model.generate_stream(
        text=args.text,
        audio_prompt_path=args.audio_prompt,
        chunk_size=args.chunk_size,
        context_window=args.context_window,
    ):
        audio_chunks.append(audio_chunk)

        # Print metrics as chunks arrive
        if metrics.latency_to_first_chunk and metrics.chunk_count == 1:
            print(f"First chunk latency: {metrics.latency_to_first_chunk:.3f}s")
        print(f"Chunk {metrics.chunk_count}: {audio_chunk.shape[-1] / model.sr:.2f}s of audio")

    # Combine all chunks
    final_audio = torch.cat(audio_chunks, dim=-1)

    # Print final metrics
    print("-" * 50)
    if metrics.total_generation_time:
        print(f"Total generation time: {metrics.total_generation_time:.3f}s")
    if metrics.total_audio_duration:
        print(f"Total audio duration: {metrics.total_audio_duration:.3f}s")
    if metrics.rtf:
        print(f"Real-time factor: {metrics.rtf:.3f} {'(real-time capable!)' if metrics.rtf < 1.0 else ''}")

    # Save output
    audio_np = final_audio.squeeze().cpu().numpy()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    wavfile.write(args.output, model.sr, audio_int16)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
