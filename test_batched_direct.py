"""
Test: Direct Batched Generation API

Tests the generate_stream_batched() method directly without the Rails layer.
This validates the core batched inference works correctly.
"""

import sys
sys.path.insert(0, "src")

import time
import numpy as np
import torch
from scipy.io import wavfile
from chatterbox import ChatterboxTurboTTS, Conditionals


def main():
    print("=" * 70)
    print("Test: Direct Batched Generation API")
    print("=" * 70)

    # Initialize
    print("\n[1] Loading model...")
    model = ChatterboxTurboTTS.from_pretrained(device="cuda")

    print("\n[2] Compiling...")
    model.compile()

    # Warmup
    print("\n[3] Warming up...")
    for _ in model.generate_stream("Hello."):
        pass

    # Prepare voice conditioning
    print("\n[4] Preparing voice conditioning...")
    model.prepare_conditionals("voice_ref.wav", exaggeration=0.5)
    conds_1 = model.conds

    # Use same conditioning for second voice (for demo)
    model.prepare_conditionals("voice_ref.wav", exaggeration=0.5)
    conds_2 = model.conds

    # Test texts
    texts = [
        "Hello! I am the first speaker. This is my sentence.",
        "Hi there! I am the second speaker. Watch us speak together!",
    ]
    conds_list = [conds_1, conds_2]

    print(f"\n[5] Running batched generation for {len(texts)} texts...")
    print(f"  Text 1: {texts[0]}")
    print(f"  Text 2: {texts[1]}")

    # Collect results per sequence
    audio_per_seq = {0: [], 1: []}
    chunk_times_per_seq = {0: [], 1: []}

    start_time = time.time()

    for chunk_results in model.generate_stream_batched(
        texts=texts,
        conds_list=conds_list,
        temperature=0.3,
    ):
        elapsed = time.time() - start_time
        for result in chunk_results:
            duration = result.audio.shape[-1] / 24000
            audio_per_seq[result.seq_id].append(result.audio)
            chunk_times_per_seq[result.seq_id].append(elapsed)
            print(f"  [Seq {result.seq_id}] chunk {len(audio_per_seq[result.seq_id])}: "
                  f"{duration:.2f}s @ t={elapsed:.2f}s")

    total_time = time.time() - start_time

    # Analyze timing
    print("\n[6] Timing Analysis:")
    for seq_id in [0, 1]:
        if chunk_times_per_seq[seq_id]:
            first = chunk_times_per_seq[seq_id][0]
            last = chunk_times_per_seq[seq_id][-1]
            print(f"  Sequence {seq_id}: first chunk @ {first:.2f}s, last @ {last:.2f}s")

    if chunk_times_per_seq[0] and chunk_times_per_seq[1]:
        diff = abs(chunk_times_per_seq[0][0] - chunk_times_per_seq[1][0])
        print(f"  First chunk time difference: {diff:.3f}s")
        if diff < 0.1:
            print("  => PARALLEL: Both sequences started within 100ms!")
        elif diff < 0.5:
            print("  => NEARLY PARALLEL: Within 500ms")
        else:
            print("  => SEQUENTIAL: Significant delay between sequences")

    # Save audio
    print("\n[7] Saving audio files...")
    for seq_id, chunks in audio_per_seq.items():
        if chunks:
            audio = torch.cat(chunks, dim=-1).squeeze().cpu().numpy()
            filename = f"output_batched_seq{seq_id}.wav"
            wavfile.write(filename, 24000, (audio * 32767).astype(np.int16))
            print(f"  Sequence {seq_id}: {len(audio)/24000:.2f}s -> {filename}")

    # Summary
    print(f"\n" + "=" * 70)
    print("Summary:")
    print(f"  Total wall time: {total_time:.2f}s")

    total_audio = sum(
        sum(c.shape[-1] for c in chunks) / 24000
        for chunks in audio_per_seq.values()
    )
    print(f"  Total audio generated: {total_audio:.2f}s")
    print(f"  Real-time factor: {total_audio / total_time:.2f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
