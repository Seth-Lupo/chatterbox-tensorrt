#!/usr/bin/env python3
"""
Test script for ChatterboxTurboTRT - Full TTS Pipeline with Baked Voice

This script tests the complete TTS pipeline with TensorRT-LLM compatible
architecture using baked voice conditioning.

Usage:
    python test_tts_turbo_trt.py --voice /path/to/reference.wav --output /path/to/output.wav

Requirements:
    - Reference audio file (>5 seconds, .wav format)
    - HuggingFace token for model download (set HF_TOKEN env var)
"""

import argparse
import logging
import time
from pathlib import Path

import torch
import numpy as np
from scipy.io import wavfile as scipy_wav

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_model_creation(device: str, voice_path: str):
    """Test model loading with baked voice."""
    logger.info("=" * 60)
    logger.info("Test 1: Model Creation with Baked Voice")
    logger.info("=" * 60)

    from src.chatterbox.tts_turbo_trt import ChatterboxTurboTRT

    logger.info(f"Loading model on {device} with voice: {voice_path}")
    start = time.time()

    model = ChatterboxTurboTRT.from_pretrained(
        device=device,
        voice_audio_path=voice_path,
    )

    elapsed = time.time() - start
    logger.info(f"Model loaded in {elapsed:.2f}s")
    logger.info(f"Voice prefix shape: {model.t3._voice_prefix.shape}")
    logger.info(f"Voice prefix length: {model.t3._voice_prefix_len} tokens")

    return model


def test_generation(model, text: str):
    """Test non-streaming generation."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Non-Streaming Generation")
    logger.info("=" * 60)

    logger.info(f"Generating: '{text}'")
    start = time.time()

    audio = model.generate(
        text=text,
        temperature=0.8,
        top_k=1000,
        top_p=0.95,
        repetition_penalty=1.2,
    )

    elapsed = time.time() - start
    duration = audio.shape[1] / model.sr

    logger.info(f"Generated in {elapsed:.2f}s")
    logger.info(f"Audio shape: {audio.shape}")
    logger.info(f"Audio duration: {duration:.2f}s")
    logger.info(f"Real-time factor: {elapsed / duration:.2f}x")

    return audio


def test_streaming(model, text: str):
    """Test streaming generation."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: Streaming Generation")
    logger.info("=" * 60)

    logger.info(f"Streaming: '{text}'")
    start = time.time()

    chunks = []
    first_chunk_time = None

    for i, (chunk, metrics) in enumerate(model.generate_stream(text=text)):
        if first_chunk_time is None:
            first_chunk_time = time.time() - start
        chunks.append(chunk)
        logger.info(f"  Chunk {i}: {chunk.shape[1]} samples, latency={metrics.latency_to_first_chunk:.3f}s" if metrics.latency_to_first_chunk else f"  Chunk {i}: {chunk.shape[1]} samples")

    elapsed = time.time() - start
    total_audio = torch.cat(chunks, dim=1) if chunks else torch.zeros(1, 0)
    duration = total_audio.shape[1] / model.sr

    logger.info(f"Streamed in {elapsed:.2f}s")
    logger.info(f"Time to first chunk: {first_chunk_time:.3f}s")
    logger.info(f"Total chunks: {len(chunks)}")
    logger.info(f"Total audio: {duration:.2f}s")

    return total_audio


def test_export(model, output_dir: str):
    """Test TensorRT-LLM export."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 4: TensorRT-LLM Export")
    logger.info("=" * 60)

    export_path = model.export_for_trtllm(output_dir)

    # Verify files
    import json
    import numpy as np

    config_path = Path(export_path) / "config.json"
    prompt_path = Path(export_path) / "prompt_table.npz"

    with open(config_path) as f:
        config = json.load(f)
    logger.info(f"Config: {config}")

    prompt_data = np.load(prompt_path)
    logger.info(f"Prompt table shape: {prompt_data['prompt_embedding_table'].shape}")
    logger.info(f"Prompt length: {prompt_data['prompt_lengths']}")

    return export_path


def main():
    parser = argparse.ArgumentParser(description="Test ChatterboxTurboTRT")
    parser.add_argument(
        "--voice", "-v",
        type=str,
        required=True,
        help="Path to reference voice audio (>5 seconds, .wav)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output.wav",
        help="Output audio file path",
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        default="Hello! This is a test of the Chatterbox Turbo TTS system with baked voice conditioning.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda, cpu, mps)",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default=None,
        help="Optional: Export TRT-LLM checkpoint to this directory",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ChatterboxTurboTRT Full Pipeline Test")
    logger.info("=" * 60)
    logger.info(f"Device: {args.device}")
    logger.info(f"Voice: {args.voice}")
    logger.info(f"Output: {args.output}")

    # Test 1: Model creation
    model = test_model_creation(args.device, args.voice)

    # Test 2: Non-streaming generation
    audio = test_generation(model, args.text)

    # Save output using scipy (avoids torchaudio/ffmpeg dependency)
    audio_np = audio.squeeze().cpu().numpy()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    scipy_wav.write(args.output, model.sr, audio_int16)
    logger.info(f"\nSaved audio to: {args.output}")

    # Test 3: Streaming
    stream_audio = test_streaming(model, args.text)

    # Test 4: Export (optional)
    if args.export_dir:
        test_export(model, args.export_dir)

    logger.info("\n" + "=" * 60)
    logger.info("All tests passed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
