#!/usr/bin/env python3
"""
Benchmark: TensorRT-LLM for T3 (GPT-2 based model)

This script attempts to:
1. Convert T3 transformer to TensorRT-LLM format
2. Build a TensorRT-LLM engine
3. Benchmark inference speed

Usage:
    python benchmark_trtllm.py
"""

import os
import sys
import time
import tempfile
from pathlib import Path

import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def check_trtllm():
    """Check TensorRT-LLM availability and version."""
    print("Checking TensorRT-LLM...")
    try:
        import tensorrt_llm
        print(f"  TensorRT-LLM version: {tensorrt_llm.__version__}")

        # Check what models are available
        from tensorrt_llm import models
        available_models = [m for m in dir(models) if not m.startswith('_')]
        print(f"  Available models: {available_models[:10]}...")  # First 10

        return True
    except ImportError as e:
        print(f"  TensorRT-LLM not available: {e}")
        return False


def check_t3_architecture():
    """Examine T3 architecture to understand what we need to convert."""
    print("\nExamining T3 architecture...")

    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.t3_config import T3Config

    # Create T3 config (same as in tts_turbo.py)
    hp = T3Config(text_tokens_dict_size=50276)
    hp.llama_config_name = "GPT2_medium"
    hp.speech_tokens_dict_size = 6563
    hp.input_pos_emb = None
    hp.speech_cond_prompt_len = 375
    hp.use_perceiver_resampler = False
    hp.emotion_adv = False

    t3 = T3(hp)

    # Get the actual GPT2 config from the model
    gpt_cfg = t3.cfg

    print(f"  T3 config:")
    print(f"    hidden_size: {gpt_cfg.hidden_size}")
    print(f"    num_hidden_layers: {gpt_cfg.num_hidden_layers}")
    print(f"    num_attention_heads: {gpt_cfg.num_attention_heads}")
    print(f"    vocab_size (speech): {hp.speech_tokens_dict_size}")

    print(f"\n  T3 components:")
    print(f"    tfmr (GPT2): {type(t3.tfmr)}")
    print(f"    speech_emb: {type(t3.speech_emb)}")
    print(f"    speech_head: {type(t3.speech_head)}")
    print(f"    text_emb: {type(t3.text_emb)}")
    print(f"    cond_enc: {type(t3.cond_enc)}")

    # Check transformer structure
    print(f"\n  GPT2 transformer structure:")
    for name, module in t3.tfmr.named_children():
        print(f"    {name}: {type(module).__name__}")

    return t3, hp


def try_trtllm_gpt2_conversion():
    """Try to convert T3's GPT-2 backbone using TensorRT-LLM."""
    print("\n" + "="*60)
    print("Attempting TensorRT-LLM GPT-2 conversion...")
    print("="*60)

    try:
        import tensorrt_llm
        from tensorrt_llm.builder import Builder
        from tensorrt_llm.network import net_guard
        from tensorrt_llm.functional import Tensor
        import tensorrt as trt

        # Check available model builders
        print("\nChecking TensorRT-LLM model builders...")

        # Try different import paths based on version
        gpt2_model = None

        # Try newer API
        try:
            from tensorrt_llm.models.gpt import GPTLMHeadModel
            print("  Found: tensorrt_llm.models.gpt.GPTLMHeadModel")
            gpt2_model = "GPTLMHeadModel"
        except ImportError:
            pass

        # Try older API
        try:
            from tensorrt_llm.models import GPT2LMHeadModel
            print("  Found: tensorrt_llm.models.GPT2LMHeadModel")
            gpt2_model = "GPT2LMHeadModel"
        except ImportError:
            pass

        # Try GPTModel
        try:
            from tensorrt_llm.models.gpt import GPTModel
            print("  Found: tensorrt_llm.models.gpt.GPTModel")
        except ImportError:
            pass

        # List all available model classes
        print("\n  All model classes in tensorrt_llm.models:")
        from tensorrt_llm import models as trtllm_models
        for attr in dir(trtllm_models):
            if 'Model' in attr or 'GPT' in attr.upper():
                print(f"    - {attr}")

        if gpt2_model is None:
            print("\n  No direct GPT-2 model found.")
            print("  TensorRT-LLM may need manual model definition for custom architectures.")
            return False

        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


class GPT2TransformerWrapper(torch.nn.Module):
    """Wrapper that calls GPT2 transformer blocks directly with embeddings input."""

    def __init__(self, gpt2_model):
        super().__init__()
        # Copy the transformer components we need
        self.wpe = gpt2_model.wpe
        self.drop = gpt2_model.drop
        self.h = gpt2_model.h
        self.ln_f = gpt2_model.ln_f

    def forward(self, inputs_embeds):
        # inputs_embeds: (batch, seq_len, hidden_size)
        batch_size, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device

        # Position embeddings
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_embeds = self.wpe(position_ids)

        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        # Transformer blocks
        for block in self.h:
            outputs = block(hidden_states)
            hidden_states = outputs[0]

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        return hidden_states


def try_torch_tensorrt_for_t3():
    """Try using torch-tensorrt to compile T3 transformer."""
    print("\n" + "="*60)
    print("Attempting torch-tensorrt compilation for T3...")
    print("="*60)

    try:
        import torch_tensorrt
        print(f"  torch-tensorrt version: {torch_tensorrt.__version__}")

        # Load T3
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        print("\nLoading model...")
        model = ChatterboxTurboTTS.from_pretrained(
            device="cuda",
            dtype="float16",
            compile_mode=None,
        )

        # Get the transformer and wrap it
        tfmr = model.t3.tfmr
        print(f"\nOriginal transformer type: {type(tfmr)}")
        print(f"Hidden size: {model.t3.cfg.hidden_size}")

        # Create wrapper that works with embeddings directly
        print("\nCreating wrapper for embeddings-based input...")
        wrapped_tfmr = GPT2TransformerWrapper(tfmr).cuda().half().eval()

        hidden_size = model.t3.cfg.hidden_size
        example_input = torch.randn(1, 100, hidden_size, device="cuda", dtype=torch.float16)

        # Test wrapper works
        print("Testing wrapper...")
        with torch.no_grad():
            test_out = wrapped_tfmr(example_input)
        print(f"  Wrapper output shape: {test_out.shape}")

        print(f"\nCompiling wrapped transformer with torch-tensorrt...")
        print(f"  Input shape: {example_input.shape}")

        # Try dynamic shapes
        compiled_tfmr = torch_tensorrt.compile(
            wrapped_tfmr,
            inputs=[
                torch_tensorrt.Input(
                    min_shape=[1, 1, hidden_size],
                    opt_shape=[1, 256, hidden_size],
                    max_shape=[1, 2048, hidden_size],
                    dtype=torch.float16,
                )
            ],
            enabled_precisions={torch.float16},
            truncate_long_and_double=True,
        )

        print("  Compilation successful!")

        # Test inference
        print("\nTesting compiled model...")
        with torch.no_grad():
            # Warmup
            for _ in range(3):
                _ = compiled_tfmr(example_input)
            torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(10):
                _ = compiled_tfmr(example_input)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

        print(f"  10 forward passes: {elapsed:.3f}s ({elapsed/10*1000:.1f}ms per pass)")

        # Compare with original
        print("\nComparing with original PyTorch...")
        with torch.no_grad():
            # Warmup
            for _ in range(3):
                _ = tfmr(example_input)
            torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(10):
                _ = tfmr(example_input)
            torch.cuda.synchronize()
            elapsed_orig = time.perf_counter() - start

        print(f"  10 forward passes: {elapsed_orig:.3f}s ({elapsed_orig/10*1000:.1f}ms per pass)")

        speedup = elapsed_orig / elapsed
        print(f"\n  Speedup: {speedup:.2f}x")

        return compiled_tfmr, model

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def benchmark_full_generation(model, compiled_tfmr=None, iterations=5):
    """Benchmark full TTS generation."""
    print("\n" + "="*60)
    print("Benchmarking full TTS generation...")
    print("="*60)

    test_text = "Hello, this is a test of the text to speech system."

    # If we have a compiled transformer, swap it in
    if compiled_tfmr is not None:
        original_tfmr = model.t3.tfmr
        model.t3.tfmr = compiled_tfmr
        print("Using TensorRT-compiled transformer")
    else:
        print("Using original PyTorch transformer")

    latencies = []

    print(f"\nRunning {iterations} iterations...")
    for i in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()

        # Generate first chunk only (measure time to first audio)
        for audio_chunk, metrics in model.generate_stream(
            text=test_text,
            chunk_size=25,
            context_window=50,
        ):
            torch.cuda.synchronize()
            latency = time.perf_counter() - start
            latencies.append(latency)
            # Consume rest
            for _ in model.generate_stream(text=test_text):
                pass
            break

        print(f"  Run {i+1}: {latency:.3f}s")
        torch.cuda.empty_cache()

    # Restore original if needed
    if compiled_tfmr is not None:
        model.t3.tfmr = original_tfmr

    import statistics
    print(f"\nResults:")
    print(f"  Mean: {statistics.mean(latencies):.3f}s")
    print(f"  Min:  {min(latencies):.3f}s")
    print(f"  Max:  {max(latencies):.3f}s")

    return latencies


def main():
    print("="*60)
    print("TensorRT-LLM / torch-tensorrt Benchmark for T3")
    print("="*60)

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
    else:
        print("ERROR: CUDA not available")
        sys.exit(1)

    # Check TensorRT-LLM
    has_trtllm = check_trtllm()

    # Examine T3 architecture
    t3, hp = check_t3_architecture()
    del t3  # Free memory
    torch.cuda.empty_cache()

    # Try TensorRT-LLM conversion
    if has_trtllm:
        try_trtllm_gpt2_conversion()

    # Try torch-tensorrt
    compiled_tfmr, model = try_torch_tensorrt_for_t3()

    if model is not None:
        # Benchmark with compiled transformer
        print("\n" + "="*60)
        print("Comparing TensorRT vs PyTorch for full generation")
        print("="*60)

        print("\n--- With TensorRT-compiled transformer ---")
        trt_latencies = benchmark_full_generation(model, compiled_tfmr, iterations=5)

        print("\n--- With original PyTorch transformer ---")
        pytorch_latencies = benchmark_full_generation(model, None, iterations=5)

        import statistics
        trt_mean = statistics.mean(trt_latencies)
        pytorch_mean = statistics.mean(pytorch_latencies)

        print("\n" + "="*60)
        print("FINAL COMPARISON")
        print("="*60)
        print(f"PyTorch mean latency:  {pytorch_mean:.3f}s")
        print(f"TensorRT mean latency: {trt_mean:.3f}s")

        if trt_mean < pytorch_mean:
            print(f"TensorRT is {pytorch_mean/trt_mean:.2f}x FASTER")
        else:
            print(f"TensorRT is {trt_mean/pytorch_mean:.2f}x SLOWER")


if __name__ == "__main__":
    main()
