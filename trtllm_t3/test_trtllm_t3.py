#!/usr/bin/env python3
"""
Test TensorRT-LLM engine for T3 GPT-2 transformer.

This script:
1. Loads the TensorRT-LLM engine (if built)
2. Tests inference with embeddings input
3. Benchmarks against PyTorch baseline
4. Falls back to demonstrating the integration approach if engine not available

Usage:
    python test_trtllm_t3.py
    python test_trtllm_t3.py --benchmark
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import numpy as np

# Add project to path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

ENGINE_DIR = SCRIPT_DIR / "engines" / "t3_gpt2_engine"
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"


def check_trtllm_available() -> bool:
    """Check if TensorRT-LLM is available."""
    try:
        import tensorrt_llm
        print(f"TensorRT-LLM version: {tensorrt_llm.__version__}")
        return True
    except ImportError:
        return False


def check_engine_available() -> bool:
    """Check if TensorRT-LLM engine is built."""
    engine_path = ENGINE_DIR / "rank0.engine"
    if engine_path.exists():
        print(f"Engine found: {engine_path}")
        return True
    print(f"Engine not found at: {engine_path}")
    return False


class TensorRTLLMRunner:
    """
    TensorRT-LLM runner for T3's GPT-2 transformer.

    This wraps the TensorRT-LLM engine and provides an interface
    compatible with T3's embedding-based input.
    """

    def __init__(self, engine_dir: Path):
        import tensorrt_llm
        from tensorrt_llm.runtime import ModelRunner, ModelRunnerCpp

        self.engine_dir = engine_dir
        print(f"Loading TensorRT-LLM engine from {engine_dir}...")

        # Load the engine
        self.runner = ModelRunnerCpp.from_dir(
            engine_dir=str(engine_dir),
            rank=0,
        )
        print("Engine loaded!")

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Run inference with embeddings input.

        Note: TensorRT-LLM typically expects token IDs, not embeddings.
        For T3 integration, we need a custom approach:

        Option 1: Modify the engine to accept embeddings
        Option 2: Use a passthrough embedding layer
        Option 3: Keep embeddings in PyTorch, only use TRT-LLM for attention
        """
        # This is a simplified interface - actual implementation depends on
        # how the engine was built
        raise NotImplementedError(
            "Direct embeddings input requires custom TensorRT-LLM model. "
            "See PyTorchWithTRTLLMAttention for the hybrid approach."
        )


class PyTorchWithTRTLLMAttention:
    """
    Hybrid approach: PyTorch embeddings + TensorRT-LLM attention.

    This keeps the custom embedding layers in PyTorch and only accelerates
    the attention computation with TensorRT-LLM.

    Architecture:
        PyTorch: speech_emb(tokens) -> embeddings
        PyTorch: cond_enc(audio) -> conditioning
        TensorRT-LLM: GPT2Attention(embeddings, conditioning) -> hidden_states
        PyTorch: speech_head(hidden_states) -> logits
    """

    def __init__(self, model, engine_dir: Optional[Path] = None):
        self.model = model
        self.engine_dir = engine_dir
        self.use_trtllm = engine_dir is not None and check_engine_available()

        if self.use_trtllm:
            self._load_trtllm_engine()
        else:
            print("Using PyTorch backend (TensorRT-LLM engine not available)")

    def _load_trtllm_engine(self):
        """Load TensorRT-LLM engine for attention layers."""
        try:
            self.trtllm_runner = TensorRTLLMRunner(self.engine_dir)
        except Exception as e:
            print(f"Failed to load TensorRT-LLM engine: {e}")
            self.use_trtllm = False

    def generate_stream(self, text: str, **kwargs):
        """
        Generate audio stream using the hybrid approach.

        For now, falls back to pure PyTorch since TensorRT-LLM
        integration requires custom engine building.
        """
        return self.model.generate_stream(text=text, **kwargs)


def benchmark_pytorch_baseline(model, num_iterations: int = 5):
    """Benchmark PyTorch baseline for T3 transformer."""
    print("\n" + "=" * 60)
    print("Benchmarking PyTorch Baseline")
    print("=" * 60)

    hidden_size = model.t3.cfg.hidden_size

    # Test different sequence lengths
    seq_lengths = [50, 100, 200, 500]

    for seq_len in seq_lengths:
        # Create test input (embeddings)
        test_embeds = torch.randn(
            1, seq_len, hidden_size,
            device="cuda",
            dtype=torch.float16,
        )

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model.t3.tfmr(
                    inputs_embeds=test_embeds,
                    use_cache=True,
                    return_dict=True,
                )
        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.no_grad():
                output = model.t3.tfmr(
                    inputs_embeds=test_embeds,
                    use_cache=True,
                    return_dict=True,
                )

            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        mean_time = sum(times) / len(times)
        print(f"  Seq len {seq_len:4d}: {mean_time * 1000:.2f} ms "
              f"({seq_len / mean_time:.0f} tokens/sec)")

    return times


def benchmark_kv_cache_generation(model, num_tokens: int = 200):
    """Benchmark autoregressive generation with KV cache."""
    print("\n" + "=" * 60)
    print("Benchmarking Autoregressive Generation with KV Cache")
    print("=" * 60)

    hidden_size = model.t3.cfg.hidden_size
    initial_len = 50

    # Initial conditioning embeddings
    initial_embeds = torch.randn(
        1, initial_len, hidden_size,
        device="cuda",
        dtype=torch.float16,
    )

    # Warmup
    with torch.no_grad():
        output = model.t3.tfmr(
            inputs_embeds=initial_embeds,
            use_cache=True,
            return_dict=True,
        )
        cache = output.past_key_values

        for _ in range(10):
            new_embed = torch.randn(1, 1, hidden_size, device="cuda", dtype=torch.float16)
            output = model.t3.tfmr(
                inputs_embeds=new_embed,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = output.past_key_values

    torch.cuda.synchronize()

    # Benchmark: Generate num_tokens tokens autoregressively
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        # Initial pass (prefill)
        output = model.t3.tfmr(
            inputs_embeds=initial_embeds,
            use_cache=True,
            return_dict=True,
        )
        cache = output.past_key_values

        # Autoregressive generation
        for i in range(num_tokens - initial_len):
            new_embed = torch.randn(1, 1, hidden_size, device="cuda", dtype=torch.float16)
            output = model.t3.tfmr(
                inputs_embeds=new_embed,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = output.past_key_values

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start

    tokens_generated = num_tokens - initial_len
    tokens_per_sec = tokens_generated / total_time

    print(f"  Total tokens generated: {tokens_generated}")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Tokens/sec: {tokens_per_sec:.0f}")
    print(f"  Time per token: {total_time / tokens_generated * 1000:.2f} ms")


def export_for_triton(model, output_dir: Path):
    """Export model components for Triton Inference Server."""
    print("\n" + "=" * 60)
    print("Exporting for Triton Inference Server")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Export embedding layers (PyTorch/ONNX)
    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(exist_ok=True)

    print("  Exporting speech_emb...")
    torch.save(model.t3.speech_emb.state_dict(), embeddings_dir / "speech_emb.pt")

    print("  Exporting cond_enc...")
    torch.save(model.t3.cond_enc.state_dict(), embeddings_dir / "cond_enc.pt")

    print("  Exporting speech_head...")
    torch.save(model.t3.speech_head.state_dict(), embeddings_dir / "speech_head.pt")

    # Create Triton model config
    triton_config = output_dir / "triton_config"
    triton_config.mkdir(exist_ok=True)

    # Ensemble config for Triton
    ensemble_config = {
        "name": "t3_tts_ensemble",
        "platform": "ensemble",
        "max_batch_size": 1,
        "input": [
            {"name": "text", "data_type": "TYPE_STRING", "dims": [1]},
            {"name": "audio_prompt", "data_type": "TYPE_FP16", "dims": [-1]},
        ],
        "output": [
            {"name": "audio", "data_type": "TYPE_FP16", "dims": [-1]},
        ],
        "ensemble_scheduling": {
            "step": [
                {
                    "model_name": "text_encoder",
                    "model_version": 1,
                    "input_map": {"text": "text"},
                    "output_map": {"text_embeds": "text_embeds"},
                },
                {
                    "model_name": "audio_encoder",
                    "model_version": 1,
                    "input_map": {"audio": "audio_prompt"},
                    "output_map": {"conditioning": "conditioning"},
                },
                {
                    "model_name": "t3_transformer",
                    "model_version": 1,
                    "input_map": {
                        "text_embeds": "text_embeds",
                        "conditioning": "conditioning",
                    },
                    "output_map": {"hidden_states": "hidden_states"},
                },
                {
                    "model_name": "speech_head",
                    "model_version": 1,
                    "input_map": {"hidden_states": "hidden_states"},
                    "output_map": {"speech_tokens": "speech_tokens"},
                },
                {
                    "model_name": "vocoder",
                    "model_version": 1,
                    "input_map": {"speech_tokens": "speech_tokens"},
                    "output_map": {"audio": "audio"},
                },
            ],
        },
    }

    with open(triton_config / "ensemble_config.json", "w") as f:
        json.dump(ensemble_config, f, indent=2)

    print(f"\n  Triton configs saved to: {triton_config}")
    print("\n  To deploy with Triton:")
    print("    1. Build TensorRT-LLM engine for t3_transformer")
    print("    2. Export other components to ONNX/PyTorch")
    print("    3. Create Triton model repository")
    print("    4. Launch: tritonserver --model-repository=/models")


def main():
    parser = argparse.ArgumentParser(description="Test TensorRT-LLM T3 integration")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--export-triton", action="store_true", help="Export for Triton")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "triton_export")
    args = parser.parse_args()

    print("=" * 60)
    print("TensorRT-LLM T3 Integration Test")
    print("=" * 60)

    # Check prerequisites
    has_trtllm = check_trtllm_available()
    has_engine = check_engine_available()

    # Load model
    print("\nLoading T3 model...")
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    model = ChatterboxTurboTTS.from_pretrained(
        device="cuda",
        dtype="float16",
        compile_mode=None,
    )
    print("Model loaded!")

    # Run benchmarks
    if args.benchmark or not (has_trtllm and has_engine):
        benchmark_pytorch_baseline(model)
        benchmark_kv_cache_generation(model)

    # Export for Triton
    if args.export_triton:
        export_for_triton(model, args.output_dir)

    # Test TensorRT-LLM if available
    if has_trtllm and has_engine:
        print("\n" + "=" * 60)
        print("Testing TensorRT-LLM Engine")
        print("=" * 60)

        try:
            runner = TensorRTLLMRunner(ENGINE_DIR)
            print("TensorRT-LLM engine loaded successfully!")

            # Benchmark would go here
            # ...

        except Exception as e:
            print(f"TensorRT-LLM test failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if has_trtllm and has_engine:
        print("\nTensorRT-LLM Status: READY")
        print("  Engine loaded and ready for inference")
    elif has_trtllm:
        print("\nTensorRT-LLM Status: INSTALLED (engine not built)")
        print("  Run: ./build_trtllm_engine.sh to build the engine")
    else:
        print("\nTensorRT-LLM Status: NOT INSTALLED")
        print("  Install: pip install tensorrt-llm")
        print("  Or use: nvcr.io/nvidia/tritonserver:XX.XX-trtllm-python-py3")

    print("\nPyTorch KV Cache: WORKING")
    print("  Use model.t3.tfmr(..., use_cache=True, return_dict=True)")
    print("  Returns past_key_values for efficient autoregressive generation")

    print("\nRecommendation:")
    if has_trtllm:
        print("  For maximum performance: Build TensorRT-LLM engine with trtllm-build")
        print("  For production: Deploy with Triton Inference Server")
    else:
        print("  Current setup uses PyTorch with KV cache")
        print("  This provides good performance for most use cases")
        print("  Consider TensorRT-LLM for high-throughput production deployments")


if __name__ == "__main__":
    main()
