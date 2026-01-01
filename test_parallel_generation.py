"""
Test: True Parallel Generation with Batched Inference

Demonstrates simultaneous audio generation across multiple rails.
Both speakers generate audio AT THE SAME TIME, not interleaved.
Outputs are layered into a single combined audio file.
"""

import sys
sys.path.insert(0, "src")

import time
import threading
import logging
import numpy as np
import torch
from scipy.io import wavfile
from chatterbox import TTSRails

# Enable logging to see batch coordination
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')


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


def layer_audio(audio_dict, sample_rate=24000):
    """Layer multiple audio tracks into a single mixed output."""
    if not audio_dict:
        return None

    # Find max length
    max_len = 0
    audio_arrays = {}

    for name, chunks in audio_dict.items():
        if chunks:
            audio = torch.cat(chunks, dim=-1).squeeze().cpu().numpy()
            audio_arrays[name] = audio
            max_len = max(max_len, len(audio))
            print(f"  {name}: {len(audio)} samples ({len(audio)/sample_rate:.2f}s)")

    if not audio_arrays:
        return None

    # Pad and mix
    mixed = np.zeros(max_len, dtype=np.float32)
    for name, audio in audio_arrays.items():
        padded = np.zeros(max_len, dtype=np.float32)
        padded[:len(audio)] = audio
        mixed += padded

    # Normalize to prevent clipping
    max_val = np.max(np.abs(mixed))
    if max_val > 0.95:
        mixed = mixed * (0.95 / max_val)

    return mixed


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

    # Using single sentences to ensure they get batched together
    text_a = "Hello, I am speaker A and this is my complete sentence for testing parallel generation."
    text_b = "Hi there, I am speaker B and this is my complete sentence for testing parallel generation."

    start_time = time.time()

    # Push complete sentences to both rails at the same time
    # This ensures they hit sentence boundaries together and get batched
    rail_a.push(text_a)
    rail_b.push(text_b)

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

    # Save individual audio files and validate
    print("\n[7] Saving individual audio files...")

    for name, chunks in results.items():
        if chunks:
            audio = torch.cat(chunks, dim=-1).squeeze().cpu().numpy()
            filename = f"output_parallel_{name}.wav"
            wavfile.write(filename, 24000, (audio * 32767).astype(np.int16))

            # Audio validation
            duration = len(audio) / 24000
            max_val = np.max(np.abs(audio))
            mean_val = np.mean(np.abs(audio))
            print(f"  Speaker {name}: {duration:.2f}s -> {filename}")
            print(f"    - Max amplitude: {max_val:.4f}")
            print(f"    - Mean amplitude: {mean_val:.4f}")
            print(f"    - Chunks: {len(chunks)}")

            # Check for potential issues
            if duration < 1.0:
                print(f"    - WARNING: Audio very short!")
            if max_val > 0.99:
                print(f"    - WARNING: Possible clipping!")
            if mean_val < 0.01:
                print(f"    - WARNING: Audio may be too quiet!")

    # Layer audio into combined output
    print("\n[8] Layering audio into combined output.wav...")
    mixed = layer_audio(results)
    if mixed is not None:
        wavfile.write("output.wav", 24000, (mixed * 32767).astype(np.int16))
        print(f"  Combined: {len(mixed)/24000:.2f}s -> output.wav")

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

        print(f"  Speaker A audio: {audio_a:.2f}s")
        print(f"  Speaker B audio: {audio_b:.2f}s")
        print(f"  Total audio generated: {total_audio:.2f}s")

        if total_time > 0:
            speedup = total_audio / total_time
            print(f"  Real-time factor: {speedup:.2f}x (higher is better)")

            if speedup > 1.5:
                print("  => EFFICIENT PARALLEL: Generated more audio than wall time!")

    print("=" * 70)
    print("\nOutput files:")
    print("  - output_parallel_A.wav (Speaker A only)")
    print("  - output_parallel_B.wav (Speaker B only)")
    print("  - output.wav (Both speakers layered together)")

    tts.shutdown()


if __name__ == "__main__":
    main()
