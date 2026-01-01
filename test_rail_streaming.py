"""
Test: Single Rail with Streaming Token Input

Simulates an LLM streaming tokens into the rail.
Text is buffered until sentence boundary or timeout, then generated as one utterance.
"""

import sys
sys.path.insert(0, "src")

import time
import threading
import numpy as np
import torch
from scipy.io import wavfile
from chatterbox import TTSRails


def simulate_llm_stream(rail, text: str, delay: float = 0.03):
    """Simulate an LLM streaming tokens with delays."""
    words = text.split()
    for i, word in enumerate(words):
        token = f" {word}" if i > 0 else word
        rail.push(token)
        print(f"  → '{token}'")
        time.sleep(delay)
    print("  [LLM done]")


def main():
    print("=" * 60)
    print("Test: Rail with Streaming Input (Sentence Buffering)")
    print("=" * 60)

    # Initialize
    print("\n[1] Initializing TTSRails...")
    tts = TTSRails(device="cuda")

    # Register voice
    print("\n[2] Registering voice...")
    tts.register_voice("narrator", "voice_ref.wav")

    # Allocate rail (300ms flush timeout)
    print("\n[3] Allocating rail...")
    rail = tts.rail("test", voice="narrator", temperature=0.3, exaggeration=0.5)

    # Test text with sentence boundaries
    # Each sentence will be buffered and generated as one utterance
    test_text = "Hello there! This is a streaming test. The rail buffers text until it sees a sentence boundary. Then it generates the whole sentence at once for optimal quality."

    print("\n[4] Streaming tokens (sentences will be buffered)...")
    llm_thread = threading.Thread(
        target=simulate_llm_stream,
        args=(rail, test_text, 0.03),  # 30ms between words
        daemon=True
    )
    llm_thread.start()

    # Collect audio
    print("\n[5] Collecting audio...")
    chunks = []
    start_time = time.time()
    sentence_count = 0

    while True:
        chunk = rail.read(timeout=0.5)

        if chunk is not None:
            chunks.append(chunk)
            elapsed = time.time() - start_time
            duration = chunk.shape[-1] / 24000

            # Detect new sentence start (chunk after a gap or first chunk)
            if len(chunks) == 1 or duration > 0.3:
                sentence_count += 1

            print(f"  Chunk {len(chunks)}: {duration:.2f}s @ {elapsed:.2f}s")

        # Stop when LLM done + rail idle + no more audio
        if not llm_thread.is_alive() and rail.is_idle and chunk is None:
            break

    # Save
    print("\n[6] Saving...")
    if chunks:
        audio = torch.cat(chunks, dim=-1).squeeze().cpu().numpy()
        wavfile.write("output.wav", 24000, (audio * 32767).astype(np.int16))

        total_duration = len(audio) / 24000
        total_time = time.time() - start_time

        print(f"\n" + "=" * 60)
        print(f"Results:")
        print(f"  Sentences:      ~{test_text.count('.') + test_text.count('!')}")
        print(f"  Audio duration: {total_duration:.2f}s")
        print(f"  Total time:     {total_time:.2f}s")
        print(f"  Chunks:         {len(chunks)}")
        print(f"  Saved to:       output.wav")
        print("=" * 60)
    else:
        print("  No audio generated!")

    tts.shutdown()


if __name__ == "__main__":
    main()
