"""
Test: True Parallel Generation with Batched Inference

Demonstrates simultaneous audio generation across multiple rails.
Both speakers generate audio AT THE SAME TIME, not interleaved.
"""

import sys
sys.path.insert(0, "src")

import time
import threading
import numpy as np
import torch
from scipy.io import wavfile
from chatterbox import TTSRails


def collect_audio(rail, name, results, timestamps):
    """Collect audio from a rail into results dict, tracking timestamps."""
    chunks = []
    chunk_times = []
    start = time.time()

    while True:
        chunk = rail.read(timeout=1.0)
        if chunk is not None:
            elapsed = time.time() - start
            chunks.append(chunk)
            chunk_times.append(elapsed)
            duration = chunk.shape[-1] / 24000
            print(f"  [{name}] chunk {len(chunks)}: {duration:.2f}s @ t={elapsed:.2f}s")
        elif rail.is_idle:
            break

    results[name] = chunks
    timestamps[name] = chunk_times


def main():
    print("=" * 70)
    print("Test: TRUE PARALLEL GENERATION (Batched Inference)")
    print("=" * 70)
    print()
    print("Both speakers will generate audio SIMULTANEOUSLY, not sequentially.")
    print("Watch the timestamps - chunks should appear at similar times!")
    print()

    # Initialize
    print("[1] Initializing TTSRails...")
    tts = TTSRails(device="cuda")

    # Register voices
    print("\n[2] Registering voices...")
    tts.register_voice("speaker_a", "voice_ref.wav")
    tts.register_voice("speaker_b", "voice_ref.wav")  # Use same file for demo

    # Allocate rails
    print("\n[3] Allocating rails...")
    rail_a = tts.rail("speaker_a", voice="speaker_a", temperature=0.3)
    rail_b = tts.rail("speaker_b", voice="speaker_b", temperature=0.3)

    # Push text to BOTH rails simultaneously
    print("\n[4] Pushing text to both rails simultaneously...")

    text_a = "Hello! I am speaker A. This is the first sentence. And here is a second one."
    text_b = "Hi there! I am speaker B. Watch how we speak together. This is amazing."

    start_time = time.time()

    # Push all text at once to both rails
    for word in text_a.split():
        rail_a.push(word + " ")
    for word in text_b.split():
        rail_b.push(word + " ")

    print(f"  Speaker A: {len(text_a)} chars")
    print(f"  Speaker B: {len(text_b)} chars")

    # Collect audio from both rails in parallel threads
    print("\n[5] Collecting audio from both rails (watch timestamps!)...")
    results = {}
    timestamps = {}

    thread_a = threading.Thread(target=collect_audio, args=(rail_a, "A", results, timestamps))
    thread_b = threading.Thread(target=collect_audio, args=(rail_b, "B", results, timestamps))

    thread_a.start()
    thread_b.start()

    thread_a.join()
    thread_b.join()

    total_time = time.time() - start_time

    # Analyze parallel execution
    print("\n[6] Analyzing parallel execution...")

    if timestamps.get("A") and timestamps.get("B"):
        first_a = timestamps["A"][0] if timestamps["A"] else float('inf')
        first_b = timestamps["B"][0] if timestamps["B"] else float('inf')
        time_diff = abs(first_a - first_b)

        print(f"  First chunk A: {first_a:.2f}s")
        print(f"  First chunk B: {first_b:.2f}s")
        print(f"  Time difference: {time_diff:.2f}s")

        if time_diff < 0.5:
            print("  => TRUE PARALLEL: Both rails started generating within 0.5s!")
        else:
            print("  => SEQUENTIAL: Rails generated one after the other")

    # Save audio files
    print("\n[7] Saving audio files...")

    for name, chunks in results.items():
        if chunks:
            audio = torch.cat(chunks, dim=-1).squeeze().cpu().numpy()
            filename = f"output_parallel_{name}.wav"
            wavfile.write(filename, 24000, (audio * 32767).astype(np.int16))
            print(f"  Speaker {name}: {len(audio)/24000:.2f}s -> {filename}")

    # Summary
    print(f"\n" + "=" * 70)
    print("Results:")
    print(f"  Total wall time: {total_time:.2f}s")
    print(f"  Speaker A chunks: {len(results.get('A', []))}")
    print(f"  Speaker B chunks: {len(results.get('B', []))}")

    if results.get("A") and results.get("B"):
        audio_a = sum(c.shape[-1] for c in results["A"]) / 24000
        audio_b = sum(c.shape[-1] for c in results["B"]) / 24000
        total_audio = audio_a + audio_b
        speedup = total_audio / total_time

        print(f"  Total audio generated: {total_audio:.2f}s")
        print(f"  Real-time factor: {speedup:.2f}x (higher is better)")

        if speedup > 1.5:
            print("  => EFFICIENT PARALLEL: Generated more audio than wall time!")

    print("=" * 70)

    tts.shutdown()


if __name__ == "__main__":
    main()
