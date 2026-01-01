"""
Benchmark: Time to First Token (TTFT)

Measures latency to first audio chunk with different configurations:
- Full GPU (default)
- T3 on CPU, S3Gen on GPU (hybrid)
- Full CPU

Usage:
    python benchmark_ttft.py
    python benchmark_ttft.py --mode cpu
    python benchmark_ttft.py --mode hybrid
"""

import sys
sys.path.insert(0, "src")

import argparse
import time
import torch
import numpy as np
from pathlib import Path

from chatterbox.tts_turbo import (
    ChatterboxTurboTTS,
    DEFAULT_RAMP_SCHEDULE,
    normalize_text,
)


def load_model_standard(device: str) -> ChatterboxTurboTTS:
    """Load model with all components on same device."""
    print(f"Loading model on {device}...")
    model = ChatterboxTurboTTS.from_pretrained(device=device)
    print("Compiling...")
    model.compile()
    return model


def load_model_hybrid() -> 'HybridTTSWrapper':
    """Load model with T3 on CPU, S3Gen on GPU."""
    print("Loading model in HYBRID mode (T3=CPU, S3Gen=GPU)...")

    # Load on CPU first to get T3 on CPU
    model = ChatterboxTurboTTS.from_pretrained(device="cpu")

    # Move S3Gen to GPU
    print("Moving S3Gen to GPU...")
    model.s3gen = model.s3gen.to("cuda")

    # Compile S3Gen on GPU
    print("Compiling S3Gen...")
    model.s3gen.flow = torch.compile(model.s3gen.flow)
    model._compiled = True

    # Create wrapper that handles device transfers
    return HybridTTSWrapper(model)


class HybridTTSWrapper:
    """Wrapper for hybrid CPU/GPU execution."""

    def __init__(self, model: ChatterboxTurboTTS):
        self.model = model
        self.sr = model.sr

    def prepare_conditionals(self, wav_path: str, exaggeration: float = 0.5):
        """Prepare conditionals - T3 conds on CPU, S3Gen conds on GPU."""
        self.model.prepare_conditionals(wav_path, exaggeration)

        # Move S3Gen conditioning to GPU
        for k, v in self.model.conds.gen.items():
            if torch.is_tensor(v):
                self.model.conds.gen[k] = v.to("cuda")

    def generate_stream(self, text: str, temperature: float = 0.8, ramp_schedule=None, **kwargs):
        """Generate with hybrid execution."""
        from chatterbox.tts_turbo import normalize_text, StreamingMetrics
        import torch.nn.functional as F

        start_time = time.time()
        metrics = StreamingMetrics()

        text = normalize_text(text)
        text_tokens = self.model.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        text_tokens = text_tokens.input_ids  # CPU

        schedule = ramp_schedule if ramp_schedule else DEFAULT_RAMP_SCHEDULE
        all_tokens = torch.tensor([], dtype=torch.long)
        chunk_buffer = []
        prev_tail = None

        with torch.inference_mode():
            # T3 generation on CPU
            for token in self._stream_tokens_cpu(text_tokens, temperature, kwargs):
                chunk_buffer.append(token)

                schedule_idx = min(metrics.chunk_count, len(schedule) - 1)
                chunk_size, context, cfm_steps = schedule[schedule_idx]

                if len(chunk_buffer) >= chunk_size:
                    new_tokens = torch.cat(chunk_buffer, dim=-1).squeeze(0)

                    # Process chunk - S3Gen on GPU
                    audio, duration, success, prev_tail = self._process_chunk_hybrid(
                        new_tokens, all_tokens, context, cfm_steps, prev_tail
                    )

                    if success:
                        metrics.chunk_count += 1
                        if metrics.chunk_count == 1:
                            metrics.latency_to_first_chunk = time.time() - start_time
                        yield audio, metrics

                    all_tokens = torch.cat([all_tokens, new_tokens]) if len(all_tokens) > 0 else new_tokens
                    chunk_buffer = []

            # Flush remaining
            if chunk_buffer:
                new_tokens = torch.cat(chunk_buffer, dim=-1).squeeze(0)
                schedule_idx = min(metrics.chunk_count, len(schedule) - 1)
                _, context, cfm_steps = schedule[schedule_idx]
                audio, duration, success, prev_tail = self._process_chunk_hybrid(
                    new_tokens, all_tokens, context, cfm_steps, prev_tail
                )
                if success:
                    metrics.chunk_count += 1
                    yield audio, metrics

            if prev_tail is not None and len(prev_tail) > 0:
                yield torch.from_numpy(prev_tail).unsqueeze(0), metrics

    def _stream_tokens_cpu(self, text_tokens, temperature, kwargs):
        """Stream tokens from T3 on CPU."""
        processors = self.model._get_logits_processors(
            temperature,
            kwargs.get('top_k', 1000),
            kwargs.get('top_p', 0.95),
            kwargs.get('repetition_penalty', 1.2),
        )

        t3_cond = self.model.conds.t3  # Already on CPU
        start_token = self.model.t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])

        embeds, _ = self.model.t3.prepare_input_embeds(
            t3_cond=t3_cond, text_tokens=text_tokens, speech_tokens=start_token, cfg_weight=0.0
        )

        generated = torch.empty((1, 1001), dtype=torch.long)
        count = 0

        outputs = self.model.t3.tfmr(inputs_embeds=embeds, use_cache=True)
        kv_cache = outputs.past_key_values

        logits = self.model.t3.speech_head(outputs[0][:, -1:])
        logits = processors(start_token, logits[:, -1, :])
        token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        generated[:, count] = token.squeeze()
        count += 1
        yield token

        stop_token = self.model.t3.hp.stop_speech_token

        for _ in range(1000):
            embed = self.model.t3.speech_emb(token)
            outputs = self.model.t3.tfmr(inputs_embeds=embed, past_key_values=kv_cache, use_cache=True)
            kv_cache = outputs.past_key_values

            logits = self.model.t3.speech_head(outputs[0])
            logits = processors(generated[:, :count], logits[:, -1, :])

            if torch.all(logits == -float("inf")):
                break

            token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            if token.item() == stop_token:
                break

            generated[:, count] = token.squeeze()
            count += 1
            yield token

    def _process_chunk_hybrid(self, new_tokens, all_tokens, context_window, cfm_steps, prev_tail):
        """Process chunk with S3Gen on GPU."""
        import numpy as np

        if len(all_tokens) > 0:
            context_tokens = all_tokens[-context_window:] if len(all_tokens) > context_window else all_tokens
            tokens = torch.cat([context_tokens, new_tokens])
            context_length = len(context_tokens)
        else:
            tokens = new_tokens
            context_length = 0

        tokens = tokens[tokens < 6561]
        if len(tokens) == 0:
            return None, 0, False, prev_tail

        # Move tokens to GPU for S3Gen
        wav, _ = self.model.s3gen.inference(
            speech_tokens=tokens.to("cuda"),
            ref_dict=self.model.conds.gen,  # Already on GPU
            n_cfm_timesteps=cfm_steps,
        )
        wav = wav.squeeze(0).detach().cpu().numpy()

        # Crop context
        if context_length > 0:
            samples_per_token = len(wav) / len(tokens)
            skip = int(context_length * samples_per_token)
            audio = wav[skip:]
        else:
            audio = wav

        if len(audio) == 0:
            return None, 0, False, prev_tail

        # Crossfade
        crossfade_samples = int(20 * 24000 / 1000)
        audio = audio - np.mean(audio)

        if prev_tail is not None and crossfade_samples > 0:
            blend_len = min(crossfade_samples, len(prev_tail), len(audio))
            if blend_len > 0:
                t = np.linspace(0, np.pi / 2, blend_len, dtype=audio.dtype)
                blended = prev_tail[:blend_len] * np.cos(t) + audio[:blend_len] * np.sin(t)
                audio = np.concatenate([blended, audio[blend_len:]])

        if len(audio) > crossfade_samples:
            new_tail = audio[-crossfade_samples:].copy()
            audio = audio[:-crossfade_samples]
        else:
            new_tail = audio.copy()
            audio = np.array([], dtype=audio.dtype)

        return torch.from_numpy(audio).unsqueeze(0), len(audio) / 24000, True, new_tail


def warmup(model, voice_path: str):
    """Warmup the model."""
    print("Warming up...")
    model.prepare_conditionals(voice_path, exaggeration=0.5)

    # Run one generation
    for _ in model.generate_stream("Hello.", ramp_schedule=[(4, 0, 1)]):
        break
    print("Warmup complete")


def benchmark_ttft(
    model: ChatterboxTurboTTS,
    text: str,
    num_runs: int = 5,
    ramp_schedule=None,
) -> dict:
    """
    Benchmark time to first token/chunk.

    Returns dict with timing breakdown.
    """
    if ramp_schedule is None:
        ramp_schedule = DEFAULT_RAMP_SCHEDULE

    text = normalize_text(text)

    results = {
        "ttft_ms": [],
        "first_chunk_duration_ms": [],
        "tokens_in_first_chunk": ramp_schedule[0][0],
        "cfm_steps_first_chunk": ramp_schedule[0][2],
    }

    for run in range(num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        start = time.perf_counter()

        for chunk, metrics in model.generate_stream(
            text,
            temperature=0.3,
            ramp_schedule=ramp_schedule,
        ):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            ttft = (time.perf_counter() - start) * 1000

            chunk_duration = chunk.shape[-1] / 24000 * 1000  # ms

            results["ttft_ms"].append(ttft)
            results["first_chunk_duration_ms"].append(chunk_duration)

            print(f"  Run {run+1}: TTFT={ttft:.1f}ms, chunk={chunk_duration:.1f}ms audio")
            break  # Only measure first chunk

    # Compute stats
    results["ttft_mean_ms"] = np.mean(results["ttft_ms"])
    results["ttft_std_ms"] = np.std(results["ttft_ms"])
    results["ttft_min_ms"] = np.min(results["ttft_ms"])
    results["ttft_max_ms"] = np.max(results["ttft_ms"])
    results["chunk_duration_mean_ms"] = np.mean(results["first_chunk_duration_ms"])

    return results


def print_results(results: dict, mode: str):
    """Print formatted results."""
    print(f"\n{'='*60}")
    print(f"TTFT Benchmark Results - {mode.upper()} mode")
    print(f"{'='*60}")
    print(f"First chunk config:")
    print(f"  Tokens:     {results['tokens_in_first_chunk']}")
    print(f"  CFM steps:  {results['cfm_steps_first_chunk']}")
    print(f"\nTime to First Token (TTFT):")
    print(f"  Mean:  {results['ttft_mean_ms']:.1f} ms")
    print(f"  Std:   {results['ttft_std_ms']:.1f} ms")
    print(f"  Min:   {results['ttft_min_ms']:.1f} ms")
    print(f"  Max:   {results['ttft_max_ms']:.1f} ms")
    print(f"\nFirst chunk audio duration:")
    print(f"  Mean:  {results['chunk_duration_mean_ms']:.1f} ms")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="TTFT Benchmark")
    parser.add_argument(
        "--mode",
        choices=["gpu", "cpu", "hybrid"],
        default="gpu",
        help="Device mode: gpu (all GPU), cpu (all CPU), hybrid (T3=CPU, S3Gen=GPU)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of benchmark runs",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="voice_ref.wav",
        help="Voice reference file",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, this is a test of the time to first token latency.",
        help="Text to synthesize",
    )
    args = parser.parse_args()

    # Check voice file exists
    if not Path(args.voice).exists():
        print(f"Error: Voice file not found: {args.voice}")
        return

    # Load model based on mode
    if args.mode == "gpu":
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            args.mode = "cpu"
            model = load_model_standard("cpu")
        else:
            model = load_model_standard("cuda")
    elif args.mode == "cpu":
        model = load_model_standard("cpu")
    elif args.mode == "hybrid":
        if not torch.cuda.is_available():
            print("CUDA not available for hybrid mode, falling back to CPU")
            args.mode = "cpu"
            model = load_model_standard("cpu")
        else:
            model = load_model_hybrid()

    # Warmup
    warmup(model, args.voice)

    # Run benchmark with default schedule
    print(f"\n--- Benchmark with DEFAULT schedule ---")
    print(f"Text: '{args.text}'")
    print(f"Runs: {args.runs}")

    results = benchmark_ttft(
        model,
        args.text,
        num_runs=args.runs,
        ramp_schedule=DEFAULT_RAMP_SCHEDULE,
    )
    print_results(results, args.mode)

    # Also test with minimal first chunk for comparison
    print(f"\n--- Benchmark with MINIMAL first chunk (2 tokens, 1 CFM) ---")
    minimal_schedule = [
        (2, 0, 1),  # Ultra minimal first chunk
        (8, 2, 3),
        (16, 10, 5),
        (32, 26, 7),
    ]

    results_minimal = benchmark_ttft(
        model,
        args.text,
        num_runs=args.runs,
        ramp_schedule=minimal_schedule,
    )
    print_results(results_minimal, f"{args.mode} (minimal)")

    # Summary comparison
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"Default schedule (4 tokens, 1 CFM):  {results['ttft_mean_ms']:.1f} ms")
    print(f"Minimal schedule (2 tokens, 1 CFM):  {results_minimal['ttft_mean_ms']:.1f} ms")
    print(f"Difference: {results['ttft_mean_ms'] - results_minimal['ttft_mean_ms']:.1f} ms")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
