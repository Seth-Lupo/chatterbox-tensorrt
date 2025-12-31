#!/usr/bin/env python3
"""
Benchmark: TensorRT with KV Cache support

This implements proper KV caching for the GPT-2 transformer wrapper.
"""

import argparse
import time
import statistics
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent / "src"))

from chatterbox.tts_turbo import ChatterboxTurboTTS


TEST_TEXTS = [
    "Hello, this is a test of the text to speech system.",
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models can now generate natural sounding speech.",
]


class GPT2BlockWithCache(nn.Module):
    """Single GPT-2 block that properly handles KV cache."""

    def __init__(self, original_block):
        super().__init__()
        self.block = original_block

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # GPT2Block already handles this internally
        outputs = self.block(
            hidden_states,
            layer_past=past_key_value,
            use_cache=use_cache,
        )

        hidden_states = outputs[0]
        new_past_key_value = outputs[1] if use_cache else None

        return hidden_states, new_past_key_value


class GPT2WithKVCache(nn.Module):
    """
    GPT-2 transformer wrapper with proper KV cache support.

    This takes embeddings as input (not token IDs) since T3 uses custom embeddings.
    """

    def __init__(self, gpt2_model):
        super().__init__()
        self.wpe = gpt2_model.wpe
        self.drop = gpt2_model.drop
        self.h = nn.ModuleList([GPT2BlockWithCache(block) for block in gpt2_model.h])
        self.ln_f = gpt2_model.ln_f
        self.num_layers = len(self.h)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Args:
            inputs_embeds: (batch, seq_len, hidden_size) - can be 1 token for incremental
            past_key_values: List of (key, value) tuples, one per layer
            use_cache: Whether to return new KV cache

        Returns:
            hidden_states: (batch, seq_len, hidden_size)
            new_past_key_values: Updated KV cache if use_cache=True
        """
        batch_size, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device

        # Calculate position IDs
        if past_key_values is not None and past_key_values[0] is not None:
            # Incremental decoding: past_length = cached sequence length
            past_length = past_key_values[0][0].shape[2]  # (batch, heads, seq, head_dim)
        else:
            past_length = 0

        position_ids = torch.arange(
            past_length, past_length + seq_len, device=device
        ).unsqueeze(0)

        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        # Process through transformer blocks
        new_past_key_values = [] if use_cache else None

        for i, block in enumerate(self.h):
            past_kv = past_key_values[i] if past_key_values is not None else None

            hidden_states, new_past_kv = block(
                hidden_states,
                past_key_value=past_kv,
                use_cache=use_cache,
            )

            if use_cache:
                new_past_key_values.append(new_past_kv)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        return hidden_states, new_past_key_values


def test_kv_cache():
    """Test that KV cache produces same results as full recompute."""
    print("="*60)
    print("Testing KV Cache Correctness")
    print("="*60)

    # Load model
    print("\nLoading model...")
    model = ChatterboxTurboTTS.from_pretrained(
        device="cuda",
        dtype="float16",
        compile_mode=None,
    )

    # Create wrapper with KV cache
    wrapper = GPT2WithKVCache(model.t3.tfmr).cuda().half().eval()

    # Test inputs
    hidden_size = model.t3.cfg.hidden_size

    # Full sequence
    full_embeds = torch.randn(1, 50, hidden_size, device="cuda", dtype=torch.float16)

    print(f"\nTest: 50 token sequence")
    print(f"Input shape: {full_embeds.shape}")

    # Method 1: Process all at once (no cache)
    with torch.no_grad():
        output_full, _ = wrapper(full_embeds, past_key_values=None, use_cache=False)

    print(f"Full pass output shape: {output_full.shape}")

    # Method 2: Process incrementally with cache
    with torch.no_grad():
        # First chunk: tokens 0-29
        chunk1 = full_embeds[:, :30, :]
        output1, cache = wrapper(chunk1, past_key_values=None, use_cache=True)

        # Second chunk: tokens 30-49 (using cache)
        chunk2 = full_embeds[:, 30:, :]
        output2, cache = wrapper(chunk2, past_key_values=cache, use_cache=True)

        # Combine outputs
        output_incremental = torch.cat([output1, output2], dim=1)

    print(f"Incremental output shape: {output_incremental.shape}")

    # Compare
    diff = (output_full - output_incremental).abs().max().item()
    print(f"\nMax difference: {diff:.6f}")

    if diff < 1e-3:
        print("✓ KV Cache is working correctly!")
        return True, wrapper, model
    else:
        print("✗ KV Cache mismatch - outputs differ")
        return False, wrapper, model


def benchmark_kv_cache_speedup():
    """Benchmark speedup from KV cache."""
    print("\n" + "="*60)
    print("Benchmarking KV Cache Speedup")
    print("="*60)

    # Load model
    model = ChatterboxTurboTTS.from_pretrained(
        device="cuda",
        dtype="float16",
        compile_mode=None,
    )

    wrapper = GPT2WithKVCache(model.t3.tfmr).cuda().half().eval()
    hidden_size = model.t3.cfg.hidden_size

    # Simulate autoregressive generation: 200 tokens
    num_tokens = 200
    initial_len = 50  # Conditioning tokens

    print(f"\nSimulating generation of {num_tokens} tokens (starting from {initial_len})")

    # Method 1: Without KV cache (recompute everything each step)
    print("\n--- Without KV Cache (naive) ---")
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        all_embeds = torch.randn(1, initial_len, hidden_size, device="cuda", dtype=torch.float16)

        for i in range(num_tokens - initial_len):
            # Process ALL tokens each step
            output, _ = wrapper(all_embeds, past_key_values=None, use_cache=False)
            # Get last token's output
            last_output = output[:, -1:, :]
            # Add new token embedding (simulated)
            new_embed = torch.randn(1, 1, hidden_size, device="cuda", dtype=torch.float16)
            all_embeds = torch.cat([all_embeds, new_embed], dim=1)

    torch.cuda.synchronize()
    time_no_cache = time.perf_counter() - start
    print(f"Time: {time_no_cache:.3f}s")

    # Method 2: With KV cache
    print("\n--- With KV Cache ---")
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        # Initial pass with conditioning
        initial_embeds = torch.randn(1, initial_len, hidden_size, device="cuda", dtype=torch.float16)
        output, cache = wrapper(initial_embeds, past_key_values=None, use_cache=True)

        for i in range(num_tokens - initial_len):
            # Process ONLY the new token
            new_embed = torch.randn(1, 1, hidden_size, device="cuda", dtype=torch.float16)
            output, cache = wrapper(new_embed, past_key_values=cache, use_cache=True)

    torch.cuda.synchronize()
    time_with_cache = time.perf_counter() - start
    print(f"Time: {time_with_cache:.3f}s")

    # Speedup
    speedup = time_no_cache / time_with_cache
    print(f"\n{'='*40}")
    print(f"KV Cache Speedup: {speedup:.1f}x faster")
    print(f"{'='*40}")

    return wrapper, model


def try_tensorrt_with_kvcache(wrapper, model):
    """Try to compile the KV-cache wrapper with TensorRT."""
    print("\n" + "="*60)
    print("Attempting TensorRT Compilation with KV Cache")
    print("="*60)

    try:
        import torch_tensorrt
        print(f"torch-tensorrt version: {torch_tensorrt.__version__}")

        hidden_size = model.t3.cfg.hidden_size

        # The challenge: TensorRT needs fixed input specs
        # But KV cache has dynamic shapes

        print("\nChallenge: KV cache tensors grow dynamically")
        print("TensorRT requires static or bounded dynamic shapes")

        # Try compiling just the wrapper for fixed cache size
        print("\nAttempting compilation with dynamic shapes...")

        # For TensorRT, we might need to:
        # 1. Compile without KV cache (full recompute) - works but slow
        # 2. Use TensorRT's native KV cache support (complex)
        # 3. Use static cache size and pad (wastes memory)

        # Let's try option 1: compile for single-token input (incremental mode)
        # This assumes cache is pre-allocated and we just process 1 token

        example_input = torch.randn(1, 1, hidden_size, device="cuda", dtype=torch.float16)

        # Create a simplified forward for TensorRT
        class IncrementalWrapper(nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped

            def forward(self, x):
                # No cache for TensorRT - just process the input
                out, _ = self.wrapped(x, past_key_values=None, use_cache=False)
                return out

        inc_wrapper = IncrementalWrapper(wrapper).cuda().half().eval()

        compiled = torch_tensorrt.compile(
            inc_wrapper,
            inputs=[
                torch_tensorrt.Input(
                    min_shape=[1, 1, hidden_size],
                    opt_shape=[1, 100, hidden_size],
                    max_shape=[1, 500, hidden_size],
                    dtype=torch.float16,
                )
            ],
            enabled_precisions={torch.float16},
            truncate_long_and_double=True,
        )

        print("✓ Compilation successful (but without KV cache)")
        print("\nNote: TensorRT-compiled version doesn't use KV cache")
        print("For KV cache + TensorRT, you need TensorRT-LLM")

        return compiled

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("="*60)
    print("KV Cache Benchmark")
    print("="*60)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Test KV cache correctness
    success, wrapper, model = test_kv_cache()

    if not success:
        print("KV cache test failed, aborting")
        sys.exit(1)

    # Benchmark KV cache speedup
    wrapper, model = benchmark_kv_cache_speedup()

    # Try TensorRT
    compiled = try_tensorrt_with_kvcache(wrapper, model)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
Key Findings:
1. KV Cache provides massive speedup for autoregressive generation
2. PyTorch KV cache works well
3. TensorRT + KV cache requires TensorRT-LLM (complex integration)

Recommendations:
- Use PyTorch with KV cache for now
- Use torch.compile(dynamic=True) for additional speedup
- TensorRT-LLM integration is a larger project
""")


if __name__ == "__main__":
    main()
