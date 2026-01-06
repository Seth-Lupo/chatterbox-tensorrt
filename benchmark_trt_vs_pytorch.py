#!/usr/bin/env python3
"""
Benchmark: TensorRT-LLM vs PyTorch for T3 TTS

Compares inference speed between:
1. Full PyTorch (ChatterboxTurboTRT)
2. TensorRT-LLM engine (unified embedding)

Saves all generated audio to a directory for quality comparison.

Usage:
    python benchmark_trt_vs_pytorch.py \
        --engine_dir ./t3_engine_unified \
        --export_dir ./t3_export_unified \
        --voice_ref voice_ref.wav \
        --runs 5 \
        --audio_dir ./benchmark_audio
"""

import argparse
import gc
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile as scipy_wav

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Test sentences of varying lengths
TEST_SENTENCES = [
    # Short
    "Hello world.",
    # Medium
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    # Long
    "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat.",
    # Very long
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair.",
]


class BenchmarkResult:
    def __init__(self, name: str):
        self.name = name
        self.times = []
        self.tokens_generated = []
        self.audio_durations = []

    def add_run(self, time_s: float, tokens: int, audio_duration_s: float):
        self.times.append(time_s)
        self.tokens_generated.append(tokens)
        self.audio_durations.append(audio_duration_s)

    def summary(self) -> dict:
        if not self.times:
            return {}
        return {
            "name": self.name,
            "runs": len(self.times),
            "avg_time_s": np.mean(self.times),
            "std_time_s": np.std(self.times),
            "min_time_s": np.min(self.times),
            "max_time_s": np.max(self.times),
            "avg_tokens": np.mean(self.tokens_generated),
            "avg_tokens_per_sec": np.mean(self.tokens_generated) / np.mean(self.times),
            "avg_audio_duration_s": np.mean(self.audio_durations),
            "avg_rtf": np.mean(self.times) / np.mean(self.audio_durations) if np.mean(self.audio_durations) > 0 else 0,
        }


def save_audio(audio_tensor: torch.Tensor, path: str, sample_rate: int = 24000):
    """Save audio tensor to WAV file."""
    audio_np = audio_tensor.squeeze().cpu().numpy()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    scipy_wav.write(path, sample_rate, audio_int16)


def benchmark_pytorch(
    voice_ref: str,
    sentences: list,
    num_runs: int,
    warmup_runs: int = 2,
    audio_dir: str = None,
) -> dict:
    """Benchmark PyTorch inference."""
    from src.chatterbox.tts_turbo_trt import ChatterboxTurboTRT

    logger.info("=" * 60)
    logger.info("PYTORCH BENCHMARK")
    logger.info("=" * 60)

    # Create audio output directory
    if audio_dir:
        pytorch_audio_dir = Path(audio_dir) / "pytorch"
        pytorch_audio_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving audio to {pytorch_audio_dir}")

    # Load model
    logger.info("Loading PyTorch model...")
    load_start = time.time()
    model = ChatterboxTurboTRT.from_pretrained(
        device="cuda",
        voice_audio_path=voice_ref,
    )
    load_time = time.time() - load_start
    logger.info(f"Model loaded in {load_time:.2f}s")

    results = {}

    for i, sentence in enumerate(sentences):
        result = BenchmarkResult(f"pytorch_sentence_{i}")

        logger.info(f"\nSentence {i}: '{sentence[:50]}...' ({len(sentence)} chars)")

        # Warmup
        logger.info(f"Warmup ({warmup_runs} runs)...")
        for _ in range(warmup_runs):
            with torch.inference_mode():
                _ = model.generate(sentence, max_tokens=500)
            torch.cuda.synchronize()

        # Benchmark runs
        logger.info(f"Benchmarking ({num_runs} runs)...")
        for run in range(num_runs):
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

            start = time.time()
            with torch.inference_mode():
                audio = model.generate(sentence, max_tokens=500)
            torch.cuda.synchronize()
            elapsed = time.time() - start

            # Calculate metrics
            audio_samples = audio.shape[-1]
            audio_duration = audio_samples / 24000  # 24kHz

            # Estimate tokens (rough - actual count requires model internals)
            tokens = int(audio_duration * 25)  # ~25 tokens/sec audio

            result.add_run(elapsed, tokens, audio_duration)
            logger.info(f"  Run {run+1}: {elapsed:.3f}s, ~{tokens} tokens, {audio_duration:.2f}s audio, RTF={elapsed/audio_duration:.3f}")

            # Save audio (last run of each sentence)
            if audio_dir and run == num_runs - 1:
                audio_path = pytorch_audio_dir / f"sentence_{i}_run_{run+1}.wav"
                save_audio(audio, str(audio_path))
                logger.info(f"  Saved: {audio_path}")

        results[f"sentence_{i}"] = result.summary()

    # Save sentences reference
    if audio_dir:
        with open(pytorch_audio_dir / "sentences.txt", "w") as f:
            for i, s in enumerate(sentences):
                f.write(f"sentence_{i}: {s}\n")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


def benchmark_trtllm(
    engine_dir: str,
    export_dir: str,
    voice_ref: str,
    sentences: list,
    num_runs: int,
    warmup_runs: int = 2,
    audio_dir: str = None,
) -> dict:
    """Benchmark TensorRT-LLM inference."""
    from test_trt_inference import T3TRTInference
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from src.chatterbox.models.s3gen import S3Gen, S3GEN_SR
    from src.chatterbox.models.s3gen.const import S3GEN_SIL
    from scipy import signal as scipy_signal

    logger.info("=" * 60)
    logger.info("TENSORRT-LLM BENCHMARK")
    logger.info("=" * 60)

    # Create audio output directory
    if audio_dir:
        trtllm_audio_dir = Path(audio_dir) / "trtllm"
        trtllm_audio_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving audio to {trtllm_audio_dir}")

    # Load components
    logger.info("Loading TRT-LLM engine...")
    load_start = time.time()

    trt_model = T3TRTInference(engine_dir, export_dir)

    # Load tokenizer
    local_path = snapshot_download(
        repo_id="ResembleAI/chatterbox-turbo",
        token=os.getenv("HF_TOKEN") or True,
        allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"]
    )
    tokenizer = AutoTokenizer.from_pretrained(local_path)

    # Load S3Gen vocoder
    s3gen = S3Gen(meanflow=True)
    s3gen.load_state_dict(load_file(Path(local_path) / "s3gen_meanflow.safetensors"))
    s3gen.to("cuda").eval()

    # Load S3Gen reference
    orig_sr, wav_np = scipy_wav.read(voice_ref)
    if wav_np.dtype == np.int16:
        wav_np = wav_np.astype(np.float32) / 32768.0
    elif wav_np.dtype == np.int32:
        wav_np = wav_np.astype(np.float32) / 2147483648.0
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=1)
    if orig_sr != S3GEN_SR:
        num_samples = int(len(wav_np) * S3GEN_SR / orig_sr)
        wav_np = scipy_signal.resample(wav_np, num_samples)
    wav = torch.from_numpy(wav_np.astype(np.float32)).cuda()
    s3gen_ref = s3gen.embed_ref(wav[:10 * S3GEN_SR], S3GEN_SR, device="cuda")

    load_time = time.time() - load_start
    logger.info(f"Engine loaded in {load_time:.2f}s")

    results = {}

    for i, sentence in enumerate(sentences):
        result = BenchmarkResult(f"trtllm_sentence_{i}")

        logger.info(f"\nSentence {i}: '{sentence[:50]}...' ({len(sentence)} chars)")

        # Tokenize
        text_tokens = tokenizer(sentence, return_tensors="pt").input_ids.cuda()

        # Warmup
        logger.info(f"Warmup ({warmup_runs} runs)...")
        for _ in range(warmup_runs):
            with torch.inference_mode():
                speech_tokens = trt_model.generate(text_tokens, max_new_tokens=500)
            torch.cuda.synchronize()

        # Benchmark runs
        logger.info(f"Benchmarking ({num_runs} runs)...")
        for run in range(num_runs):
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

            # --- TRT-LLM Token Generation ---
            start_trt = time.time()
            with torch.inference_mode():
                speech_tokens = trt_model.generate(text_tokens, max_new_tokens=500)
            torch.cuda.synchronize()
            trt_time = time.time() - start_trt

            # --- Vocoder ---
            start_vocoder = time.time()
            speech_tokens_filtered = speech_tokens[speech_tokens < 6561]
            silence = torch.tensor([S3GEN_SIL] * 3, dtype=torch.long, device="cuda")
            speech_tokens_final = torch.cat([speech_tokens_filtered, silence])

            with torch.inference_mode():
                audio, _ = s3gen.inference(
                    speech_tokens=speech_tokens_final,
                    ref_dict=s3gen_ref,
                    n_cfm_timesteps=2,
                )
            torch.cuda.synchronize()
            vocoder_time = time.time() - start_vocoder

            total_time = trt_time + vocoder_time

            # Calculate metrics
            num_tokens = len(speech_tokens_filtered)
            audio_samples = audio.shape[-1]
            audio_duration = audio_samples / 24000

            result.add_run(total_time, num_tokens, audio_duration)
            logger.info(f"  Run {run+1}: {total_time:.3f}s (TRT={trt_time:.3f}s, vocoder={vocoder_time:.3f}s), {num_tokens} tokens, {audio_duration:.2f}s audio, RTF={total_time/audio_duration:.3f}")

            # Save audio (last run of each sentence)
            if audio_dir and run == num_runs - 1:
                audio_path = trtllm_audio_dir / f"sentence_{i}_run_{run+1}.wav"
                save_audio(audio, str(audio_path))
                logger.info(f"  Saved: {audio_path}")

        # Store additional TRT-specific metrics
        summary = result.summary()
        summary["avg_trt_time_s"] = trt_time  # Last run's TRT time (approximate)
        results[f"sentence_{i}"] = summary

    # Save sentences reference
    if audio_dir:
        with open(trtllm_audio_dir / "sentences.txt", "w") as f:
            for i, s in enumerate(sentences):
                f.write(f"sentence_{i}: {s}\n")

    return results


def print_comparison(pytorch_results: dict, trtllm_results: dict):
    """Print side-by-side comparison."""
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 80)

    print(f"\n{'Sentence':<12} {'PyTorch (s)':<14} {'TRT-LLM (s)':<14} {'Speedup':<10} {'PyTorch RTF':<12} {'TRT-LLM RTF':<12}")
    print("-" * 80)

    total_pytorch = 0
    total_trtllm = 0

    for key in pytorch_results:
        pt = pytorch_results[key]
        trt = trtllm_results[key]

        pt_time = pt["avg_time_s"]
        trt_time = trt["avg_time_s"]
        speedup = pt_time / trt_time if trt_time > 0 else 0

        total_pytorch += pt_time
        total_trtllm += trt_time

        print(f"{key:<12} {pt_time:<14.3f} {trt_time:<14.3f} {speedup:<10.2f}x {pt['avg_rtf']:<12.3f} {trt['avg_rtf']:<12.3f}")

    print("-" * 80)
    overall_speedup = total_pytorch / total_trtllm if total_trtllm > 0 else 0
    print(f"{'TOTAL':<12} {total_pytorch:<14.3f} {total_trtllm:<14.3f} {overall_speedup:<10.2f}x")

    logger.info("\nKey Metrics:")
    logger.info(f"  Overall Speedup: {overall_speedup:.2f}x")
    logger.info(f"  PyTorch avg RTF: {np.mean([r['avg_rtf'] for r in pytorch_results.values()]):.3f}")
    logger.info(f"  TRT-LLM avg RTF: {np.mean([r['avg_rtf'] for r in trtllm_results.values()]):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark TRT-LLM vs PyTorch")
    parser.add_argument("--engine_dir", type=str, default="./t3_engine_unified")
    parser.add_argument("--export_dir", type=str, default="./t3_export_unified")
    parser.add_argument("--voice_ref", type=str, default="voice_ref.wav")
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs per sentence")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup runs")
    parser.add_argument("--pytorch_only", action="store_true", help="Only run PyTorch benchmark")
    parser.add_argument("--trtllm_only", action="store_true", help="Only run TRT-LLM benchmark")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output JSON file")
    parser.add_argument("--audio_dir", type=str, default="./benchmark_audio", help="Directory to save audio files")
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("T3 TTS BENCHMARK: TensorRT-LLM vs PyTorch")
    logger.info("=" * 80)
    logger.info(f"Sentences: {len(TEST_SENTENCES)}")
    logger.info(f"Runs per sentence: {args.runs}")
    logger.info(f"Warmup runs: {args.warmup}")
    logger.info(f"Audio output: {args.audio_dir}")

    # Create audio directory
    if args.audio_dir:
        Path(args.audio_dir).mkdir(parents=True, exist_ok=True)

    pytorch_results = {}
    trtllm_results = {}

    # Run PyTorch benchmark
    if not args.trtllm_only:
        pytorch_results = benchmark_pytorch(
            voice_ref=args.voice_ref,
            sentences=TEST_SENTENCES,
            num_runs=args.runs,
            warmup_runs=args.warmup,
            audio_dir=args.audio_dir,
        )

    # Run TRT-LLM benchmark
    if not args.pytorch_only:
        trtllm_results = benchmark_trtllm(
            engine_dir=args.engine_dir,
            export_dir=args.export_dir,
            voice_ref=args.voice_ref,
            sentences=TEST_SENTENCES,
            num_runs=args.runs,
            warmup_runs=args.warmup,
            audio_dir=args.audio_dir,
        )

    # Print comparison
    if pytorch_results and trtllm_results:
        print_comparison(pytorch_results, trtllm_results)

    # Save results
    all_results = {
        "pytorch": pytorch_results,
        "trtllm": trtllm_results,
        "config": {
            "runs": args.runs,
            "warmup": args.warmup,
            "sentences": TEST_SENTENCES,
        }
    }

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
