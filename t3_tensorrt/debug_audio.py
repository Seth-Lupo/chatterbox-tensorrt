#!/usr/bin/env python3
"""
Debug audio output issues by testing different save configurations.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

# Add project to path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def debug_audio():
    """Generate audio and save with different configurations to debug distortion."""
    print("=" * 60)
    print("Audio Debug Script")
    print("=" * 60)

    # Load model
    print("\nLoading T3 model...")
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    model = ChatterboxTurboTTS.from_pretrained(
        device="cuda",
        dtype="float16",
        compile_mode=None,
    )
    print("Model loaded!")

    # Generate audio
    text = "Hello, this is a test. I love tests. Ever since I was little, I took tests like this audio one. It was amazing!"
    print(f"\nGenerating audio for: \"{text}\"")

    # Use non-streaming generation for coherent audio
    # (streaming mode has chunk boundary artifacts causing jittery intonation)
    audio = model.generate(
        text=text,
        audio_prompt_path=None,
    )
    sr = model.sr

    # Debug info
    print("\n" + "=" * 60)
    print("DEBUG INFO")
    print("=" * 60)
    print(f"Model sample rate (model.sr): {sr}")
    print(f"Audio tensor shape: {audio.shape}")
    print(f"Audio tensor dtype: {audio.dtype}")
    print(f"Audio device: {audio.device}")
    print(f"Audio min: {audio.min().item():.4f}")
    print(f"Audio max: {audio.max().item():.4f}")
    print(f"Audio mean: {audio.mean().item():.4f}")
    print(f"Audio std: {audio.std().item():.4f}")

    # Move to CPU
    audio_cpu = audio.cpu().float()
    print(f"\nAfter CPU transfer:")
    print(f"  Shape: {audio_cpu.shape}")
    print(f"  Dtype: {audio_cpu.dtype}")

    # Check dimensions
    if audio_cpu.dim() == 1:
        print("  Audio is 1D (samples only)")
        audio_1d = audio_cpu
    elif audio_cpu.dim() == 2:
        print(f"  Audio is 2D: {audio_cpu.shape}")
        if audio_cpu.shape[0] == 1:
            print("  -> Batch dim of 1, squeezing")
            audio_1d = audio_cpu.squeeze(0)
        elif audio_cpu.shape[0] == 2:
            print("  -> Might be stereo, taking first channel")
            audio_1d = audio_cpu[0]
        else:
            print(f"  -> Unknown layout, assuming first dim is batch")
            audio_1d = audio_cpu[0]
    elif audio_cpu.dim() == 3:
        print(f"  Audio is 3D: {audio_cpu.shape}")
        audio_1d = audio_cpu.squeeze(0).squeeze(0)
    else:
        print(f"  Unexpected dims: {audio_cpu.dim()}")
        audio_1d = audio_cpu.flatten()

    print(f"\nFinal 1D audio shape: {audio_1d.shape}")
    audio_np = audio_1d.numpy()

    # Calculate duration
    duration = len(audio_np) / sr
    print(f"Duration at {sr} Hz: {duration:.2f}s")

    # Test different sample rates
    test_sample_rates = [sr, 22050, 24000, 44100, 48000, 16000]

    print("\n" + "=" * 60)
    print("SAVING TEST FILES")
    print("=" * 60)

    output_dir = SCRIPT_DIR / "debug_output"
    output_dir.mkdir(exist_ok=True)

    # Normalize audio
    audio_normalized = audio_np / (np.abs(audio_np).max() + 1e-8)

    for test_sr in test_sample_rates:
        # Save as int16
        audio_int16 = (audio_normalized * 32767).astype(np.int16)
        output_path = output_dir / f"test_sr{test_sr}_int16.wav"
        wavfile.write(str(output_path), test_sr, audio_int16)
        print(f"Saved: {output_path.name} ({output_path.stat().st_size / 1024:.1f} KB)")

    # Also save float32 version at model's sample rate
    output_path = output_dir / f"test_sr{sr}_float32.wav"
    wavfile.write(str(output_path), sr, audio_normalized.astype(np.float32))
    print(f"Saved: {output_path.name} ({output_path.stat().st_size / 1024:.1f} KB)")

    # Try different normalization approaches
    print("\n" + "=" * 60)
    print("TESTING DIFFERENT NORMALIZATIONS")
    print("=" * 60)

    # 1. Raw values (no normalization)
    audio_raw = (audio_np * 32767).astype(np.int16)
    output_path = output_dir / "test_raw_no_norm.wav"
    wavfile.write(str(output_path), sr, audio_raw)
    print(f"Saved: {output_path.name} (raw, no normalization)")

    # 2. Clipped to [-1, 1]
    audio_clipped = np.clip(audio_np, -1.0, 1.0)
    audio_clipped_int16 = (audio_clipped * 32767).astype(np.int16)
    output_path = output_dir / "test_clipped.wav"
    wavfile.write(str(output_path), sr, audio_clipped_int16)
    print(f"Saved: {output_path.name} (clipped to [-1, 1])")

    # 3. Check if audio needs to be transposed (in case it's channels x samples)
    if audio_cpu.dim() == 2 and audio_cpu.shape[0] > 1:
        audio_transposed = audio_cpu.T.numpy()
        if audio_transposed.shape[0] > audio_transposed.shape[1]:
            # This is samples x channels
            audio_transposed = audio_transposed[:, 0]  # Take first channel
        audio_transposed = audio_transposed / (np.abs(audio_transposed).max() + 1e-8)
        audio_transposed_int16 = (audio_transposed * 32767).astype(np.int16)
        output_path = output_dir / "test_transposed.wav"
        wavfile.write(str(output_path), sr, audio_transposed_int16)
        print(f"Saved: {output_path.name} (transposed)")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"\nTest files saved to: {output_dir}")
    print("\nListen to each file and identify which sounds correct.")
    print("The correct one will tell us:")
    print("  - If sample rate is wrong (test_sr*.wav)")
    print("  - If normalization is wrong (test_*_norm.wav)")


if __name__ == "__main__":
    debug_audio()
