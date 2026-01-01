"""
Chatterbox Turbo Streaming TTS Example

Usage:
    python example_stream.py
    python example_stream.py --audio_prompt voice.wav
    python example_stream.py --text "Your custom text here"
"""

import sys
sys.path.insert(0, "src")

import argparse
import numpy as np
import torch
from scipy.io import wavfile
from chatterbox import ChatterboxTurboTTS


def main():
    parser = argparse.ArgumentParser(description="Chatterbox Turbo Streaming TTS")
    parser.add_argument("--text", type=str,
        default="Through silicon dreams and whispered code, I walk the paths that few have strode. A voice emerged from ones and zeros, speaking truths like digital heroes.")
    parser.add_argument("--audio_prompt", type=str, default="voice_ref.wav",
        help="Path to reference audio for voice cloning (must be >5 seconds)")
    parser.add_argument("--output", type=str, default="output.wav",
        help="Output audio file path")
    parser.add_argument("--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compile", action="store_true",
        help="Enable torch.compile for faster inference")
    args = parser.parse_args()

    print(f"Loading model on {args.device}...")
    model = ChatterboxTurboTTS.from_pretrained(device=args.device)

    if args.compile:
        print("Compiling model...")
        model.compile()
        # Warmup
        for _ in model.generate_stream("Hello."):
            pass
        print("Warmup complete.")

    print(f"\nGenerating: '{args.text}'")
    print("-" * 50)

    audio_chunks = []
    for chunk, metrics in model.generate_stream(
        text=args.text,
        audio_prompt_path=args.audio_prompt,
        temperature=0.3,
        exaggeration=0.5,
    ):
        audio_chunks.append(chunk)
        if metrics.latency_to_first_chunk and metrics.chunk_count == 1:
            print(f"First chunk latency: {metrics.latency_to_first_chunk:.3f}s")
        print(f"Chunk {metrics.chunk_count}: {chunk.shape[-1] / model.sr:.2f}s")

    # Combine and save
    final_audio = torch.cat(audio_chunks, dim=-1)
    print("-" * 50)
    if metrics.total_generation_time:
        print(f"Total time: {metrics.total_generation_time:.3f}s")
    if metrics.rtf:
        rtf_status = "(real-time!)" if metrics.rtf < 1.0 else ""
        print(f"RTF: {metrics.rtf:.3f} {rtf_status}")

    audio_np = final_audio.squeeze().cpu().numpy()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    wavfile.write(args.output, model.sr, audio_int16)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
