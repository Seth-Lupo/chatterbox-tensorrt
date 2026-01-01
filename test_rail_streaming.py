"""
Test: Single Rail with Streaming Token Input

Simulates an LLM streaming tokens into the rail while it generates audio.
"""

import sys
sys.path.insert(0, "src")

import time
import threading
import numpy as np
import torch
from scipy.io import wavfile
from chatterbox import TTSRails


def simulate_llm_stream(rail, text: str, delay: float = 0.05):
    """Simulate an LLM streaming tokens with delays."""
    words = text.split()
    for i, word in enumerate(words):
        # Add space before word (except first)
        token = f" {word}" if i > 0 else word
        rail.push(token)
        print(f"  Pushed: '{token}'")
        time.sleep(delay)
    print("  [LLM stream complete]")


def main():
    print("=" * 60)
    print("Test: Single Rail with Streaming Input")
    print("=" * 60)

    # Initialize
    print("\n[1] Initializing TTSRails...")
    tts = TTSRails(device="cuda")

    # Register voice
    print("\n[2] Registering voice...")
    tts.register_voice("narrator", "voice_ref.wav")

    # Allocate rail
    print("\n[3] Allocating rail...")
    rail = tts.rail("test", voice="narrator", temperature=0.3, exaggeration=0.5)

    # Start LLM simulation in background thread
    test_text = "Hello! This is a test of streaming text input. Each word arrives with a small delay, simulating an LLM generating tokens. The rail should append everything into one continuous utterance."

    print("\n[4] Starting LLM stream simulation...")
    llm_thread = threading.Thread(
        target=simulate_llm_stream,
        args=(rail, test_text, 0.08),  # 80ms between words
        daemon=True
    )
    llm_thread.start()

    # Collect audio while LLM is streaming
    print("\n[5] Collecting audio chunks...")
    chunks = []
    start_time = time.time()

    while True:
        chunk = rail.read(timeout=0.5)

        if chunk is not None:
            chunks.append(chunk)
            elapsed = time.time() - start_time
            duration = chunk.shape[-1] / 24000
            print(f"  Got chunk {len(chunks)}: {duration:.2f}s audio @ {elapsed:.2f}s elapsed")

        # Stop when LLM is done AND rail is idle AND no more audio
        if not llm_thread.is_alive() and rail.is_idle and chunk is None:
            break

    # Save audio
    print("\n[6] Saving audio...")
    if chunks:
        audio = torch.cat(chunks, dim=-1).squeeze().cpu().numpy()
        wavfile.write("output.wav", 24000, (audio * 32767).astype(np.int16))

        total_duration = len(audio) / 24000
        total_time = time.time() - start_time
        rtf = total_time / total_duration if total_duration > 0 else 0

        print(f"\n" + "=" * 60)
        print(f"Results:")
        print(f"  Audio duration: {total_duration:.2f}s")
        print(f"  Total time:     {total_time:.2f}s")
        print(f"  RTF:            {rtf:.2f}x")
        print(f"  Chunks:         {len(chunks)}")
        print(f"  Saved to:       output.wav")
        print("=" * 60)
    else:
        print("  No audio generated!")

    # Cleanup
    tts.shutdown()


if __name__ == "__main__":
    main()
