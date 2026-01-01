"""
Test: Two Rails Speaking Simultaneously

Demonstrates GPU lock behavior - rails interleave at sentence boundaries.
"""

import sys
sys.path.insert(0, "src")

import time
import threading
import numpy as np
import torch
from scipy.io import wavfile
from chatterbox import TTSRails


def collect_audio(rail, name, results):
    """Collect audio from a rail into results dict."""
    chunks = []
    while True:
        chunk = rail.read(timeout=1.0)
        if chunk is not None:
            chunks.append(chunk)
            print(f"  [{name}] chunk {len(chunks)}: {chunk.shape[-1]/24000:.2f}s")
        elif rail.is_idle:
            break
    results[name] = chunks


def main():
    print("=" * 60)
    print("Test: Two Rails Speaking Simultaneously")
    print("=" * 60)

    # Initialize
    print("\n[1] Initializing TTSRails...")
    tts = TTSRails(device="cuda")

    # Register voices (using same file for demo - use different files for real male/female)
    print("\n[2] Registering voices...")
    tts.register_voice("man", "voice_ref.wav")
    # If you have a second voice file:
    # tts.register_voice("woman", "voice_female.wav")
    # For now, use same voice but we'll see the interleaving behavior
    tts.register_voice("woman", "voice_ref.wav")

    # Allocate rails
    print("\n[3] Allocating rails...")
    man = tts.rail("man", voice="man", temperature=0.3)
    woman = tts.rail("woman", voice="woman", temperature=0.3)

    # Push text to BOTH at the same time
    print("\n[4] Pushing text to both rails simultaneously...")

    man_text = "Hello! I am the first speaker. This is my sentence. And here is another one."
    woman_text = "Hi there! I am the second speaker. Watch how we interleave. This is interesting."

    start_time = time.time()

    # Push all text at once (simulating simultaneous input)
    for word in man_text.split():
        man.push(word + " ")
    for word in woman_text.split():
        woman.push(word + " ")

    print(f"  Man pushed: {len(man_text)} chars")
    print(f"  Woman pushed: {len(woman_text)} chars")

    # Collect audio from both rails in parallel threads
    print("\n[5] Collecting audio from both rails...")
    results = {}

    man_thread = threading.Thread(target=collect_audio, args=(man, "man", results))
    woman_thread = threading.Thread(target=collect_audio, args=(woman, "woman", results))

    man_thread.start()
    woman_thread.start()

    man_thread.join()
    woman_thread.join()

    total_time = time.time() - start_time

    # Save audio files
    print("\n[6] Saving audio files...")

    for name, chunks in results.items():
        if chunks:
            audio = torch.cat(chunks, dim=-1).squeeze().cpu().numpy()
            filename = f"output_{name}.wav"
            wavfile.write(filename, 24000, (audio * 32767).astype(np.int16))
            print(f"  {name}: {len(audio)/24000:.2f}s -> {filename}")

    # Summary
    print(f"\n" + "=" * 60)
    print("Results:")
    print(f"  Total wall time: {total_time:.2f}s")
    print(f"  Man chunks: {len(results.get('man', []))}")
    print(f"  Woman chunks: {len(results.get('woman', []))}")
    print("=" * 60)
    print("\nNote: Rails interleave at SENTENCE boundaries, not chunks.")
    print("One rail blocks the other during each sentence generation.")

    tts.shutdown()


if __name__ == "__main__":
    main()
