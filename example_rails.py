"""
TTSRails Example - Parallel TTS Streams

Demonstrates multiple rails generating audio simultaneously,
each with their own voice.

Usage:
    python example_rails.py
"""

import sys
sys.path.insert(0, "src")

import time
import numpy as np
import torch
from scipy.io import wavfile
from chatterbox import TTSRails


def main():
    # Initialize (compiles model, warms up)
    print("Initializing TTSRails...")
    tts = TTSRails(device="cuda")

    # Register voices
    print("\nRegistering voices...")
    tts.register_voice("narrator", "voice_ref.wav")
    # tts.register_voice("assistant", "voice_assistant.wav")  # Add more as needed

    # Allocate rails
    narrator = tts.rail("narrator", voice="narrator", temperature=0.3)

    # Push text
    print("\nPushing text to narrator...")
    narrator.push("Welcome to the future of text to speech. ")
    narrator.push("This system can handle multiple voices in parallel. ")
    narrator.push("Each rail operates independently with its own voice.")

    # Collect audio
    print("\nCollecting audio...")
    chunks = []
    while True:
        chunk = narrator.read(timeout=0.5)
        if chunk is None:
            if narrator.is_idle:
                break
            continue
        chunks.append(chunk)
        print(f"  Got chunk: {chunk.shape[-1] / 24000:.2f}s")

    # Save
    if chunks:
        audio = torch.cat(chunks, dim=-1).squeeze().cpu().numpy()
        wavfile.write("output_rails.wav", 24000, (audio * 32767).astype(np.int16))
        print(f"\nSaved to output_rails.wav ({len(audio)/24000:.2f}s)")

    # Cleanup
    tts.shutdown()


def example_parallel():
    """Example with two rails running in parallel."""
    tts = TTSRails(device="cuda")

    # Register two different voices
    tts.register_voice("narrator", "voice_ref.wav")
    # tts.register_voice("character", "voice_character.wav")

    # Allocate rails
    narrator = tts.rail("narrator", voice="narrator")
    # character = tts.rail("character", voice="character")

    # Push to both (they'll generate in parallel, sharing GPU)
    narrator.push("The narrator speaks with a calm, measured tone.")
    # character.push("But I have my own voice and personality!")

    # Read from both
    narrator_chunks = []
    # character_chunks = []

    deadline = time.time() + 10  # 10 second timeout
    while time.time() < deadline:
        # Non-blocking reads
        if chunk := narrator.read_nowait():
            narrator_chunks.append(chunk)
        # if chunk := character.read_nowait():
        #     character_chunks.append(chunk)

        if narrator.is_idle:  # and character.is_idle:
            break

        time.sleep(0.01)

    # Save outputs
    if narrator_chunks:
        audio = torch.cat(narrator_chunks, dim=-1).squeeze().cpu().numpy()
        wavfile.write("narrator.wav", 24000, (audio * 32767).astype(np.int16))
        print(f"Saved narrator.wav")

    tts.shutdown()


def example_interrupt():
    """Example demonstrating interrupt."""
    tts = TTSRails(device="cuda")
    tts.register_voice("narrator", "voice_ref.wav")

    narrator = tts.rail("narrator", voice="narrator")

    # Push a long text
    narrator.push("This is a very long sentence that will take a while to generate. " * 5)

    # Wait a bit then interrupt
    time.sleep(1.0)
    print("Interrupting...")
    narrator.interrupt()

    # Rail is immediately ready for new input
    narrator.push("Short message after interrupt.")

    # Collect
    chunks = []
    while True:
        chunk = narrator.read(timeout=0.5)
        if chunk is None and narrator.is_idle:
            break
        if chunk:
            chunks.append(chunk)

    if chunks:
        audio = torch.cat(chunks, dim=-1).squeeze().cpu().numpy()
        wavfile.write("interrupted.wav", 24000, (audio * 32767).astype(np.int16))
        print(f"Saved interrupted.wav")

    tts.shutdown()


if __name__ == "__main__":
    main()
