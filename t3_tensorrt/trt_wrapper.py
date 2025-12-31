#!/usr/bin/env python3
"""
TensorRT wrapper for T3 transformer.

This provides a PyTorch-like interface to the TensorRT engine,
allowing seamless integration with T3's embedding and head layers.

Usage:
    from trt_wrapper import T3TensorRTTransformer

    trt_transformer = T3TensorRTTransformer("t3_transformer.engine")
    output = trt_transformer(input_embeds)
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tensorrt as trt


class T3TensorRTTransformer:
    """
    TensorRT-accelerated T3 transformer.

    Takes embeddings as input (just like the PyTorch version).
    Returns hidden states that can be passed to speech_head.
    """

    def __init__(self, engine_path: str, device: int = 0):
        """
        Load TensorRT engine.

        Args:
            engine_path: Path to .engine file
            device: CUDA device ID
        """
        self.engine_path = Path(engine_path)
        self.device = device

        if not self.engine_path.exists():
            raise FileNotFoundError(f"Engine not found: {engine_path}")

        # Initialize TensorRT
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        # Load engine
        print(f"Loading TensorRT engine: {engine_path}")
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")

        # Create execution context
        self.context = self.engine.create_execution_context()

        # Get input/output info
        self.input_name = "inputs_embeds"
        self.output_name = "hidden_states"

        # Get binding indices
        self.input_idx = self.engine.get_binding_index(self.input_name)
        self.output_idx = self.engine.get_binding_index(self.output_name)

        print(f"  Input binding: {self.input_name} (idx={self.input_idx})")
        print(f"  Output binding: {self.output_name} (idx={self.output_idx})")

        # Pre-allocate output buffer for common sizes
        self._output_buffer = None
        self._max_seq_len = 0

    def _ensure_output_buffer(self, batch_size: int, seq_len: int, hidden_size: int):
        """Ensure output buffer is large enough."""
        if self._output_buffer is None or seq_len > self._max_seq_len:
            self._max_seq_len = max(seq_len, 2048)  # Allocate for max
            self._output_buffer = torch.empty(
                (batch_size, self._max_seq_len, hidden_size),
                dtype=torch.float16,
                device=f"cuda:{self.device}",
            )

    def __call__(
        self,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run transformer inference.

        Args:
            inputs_embeds: (batch, seq_len, hidden_size) float16 tensor on CUDA

        Returns:
            hidden_states: (batch, seq_len, hidden_size) float16 tensor on CUDA
        """
        # Validate input
        assert inputs_embeds.is_cuda, "Input must be on CUDA"
        assert inputs_embeds.dtype == torch.float16, "Input must be float16"
        assert inputs_embeds.is_contiguous(), "Input must be contiguous"

        batch_size, seq_len, hidden_size = inputs_embeds.shape

        # Set input shape for dynamic axes
        self.context.set_binding_shape(self.input_idx, (batch_size, seq_len, hidden_size))

        # Prepare output buffer
        output = torch.empty(
            (batch_size, seq_len, hidden_size),
            dtype=torch.float16,
            device=inputs_embeds.device,
        )

        # Get data pointers
        bindings = [None, None]
        bindings[self.input_idx] = inputs_embeds.data_ptr()
        bindings[self.output_idx] = output.data_ptr()

        # Run inference
        success = self.context.execute_v2(bindings)
        if not success:
            raise RuntimeError("TensorRT inference failed")

        return output

    def benchmark(self, seq_len: int = 256, num_iterations: int = 100, warmup: int = 10):
        """
        Benchmark inference speed.

        Args:
            seq_len: Sequence length to test
            num_iterations: Number of iterations
            warmup: Number of warmup iterations

        Returns:
            dict with timing statistics
        """
        import time

        hidden_size = 1024  # T3's hidden size
        test_input = torch.randn(
            1, seq_len, hidden_size,
            dtype=torch.float16,
            device=f"cuda:{self.device}",
        )

        # Warmup
        for _ in range(warmup):
            _ = self(test_input)
        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = self(test_input)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        return {
            "seq_len": seq_len,
            "iterations": num_iterations,
            "mean_ms": np.mean(times) * 1000,
            "std_ms": np.std(times) * 1000,
            "min_ms": np.min(times) * 1000,
            "max_ms": np.max(times) * 1000,
            "throughput_tokens_per_sec": seq_len / np.mean(times),
        }


class T3HybridModel:
    """
    Hybrid T3 model: PyTorch embeddings + TensorRT transformer + PyTorch head.

    This integrates the TensorRT-accelerated transformer with T3's
    custom embedding and head layers.
    """

    def __init__(self, t3_model, engine_path: str):
        """
        Create hybrid model.

        Args:
            t3_model: Original T3 model (for embeddings and head)
            engine_path: Path to TensorRT engine
        """
        self.t3 = t3_model

        # Load TensorRT transformer
        self.trt_transformer = T3TensorRTTransformer(engine_path)

        # Keep PyTorch components
        self.speech_emb = t3_model.speech_emb
        self.text_emb = t3_model.text_emb
        self.cond_enc = t3_model.cond_enc
        self.speech_head = t3_model.speech_head

    def forward_transformer(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """
        Run transformer with TensorRT.

        Args:
            inputs_embeds: Combined embeddings from speech_emb + cond_enc

        Returns:
            hidden_states: Transformer output
        """
        # Ensure correct dtype and contiguous
        if inputs_embeds.dtype != torch.float16:
            inputs_embeds = inputs_embeds.half()
        if not inputs_embeds.is_contiguous():
            inputs_embeds = inputs_embeds.contiguous()

        return self.trt_transformer(inputs_embeds)


def load_engine(engine_path: str) -> T3TensorRTTransformer:
    """Load TensorRT engine."""
    return T3TensorRTTransformer(engine_path)


if __name__ == "__main__":
    # Quick test
    import sys

    engine_path = Path(__file__).parent / "t3_transformer.engine"

    if not engine_path.exists():
        print(f"Engine not found: {engine_path}")
        print("Run: ./build_engine.sh first")
        sys.exit(1)

    # Load and test
    trt_transformer = T3TensorRTTransformer(str(engine_path))

    # Test inference
    print("\nTesting inference...")
    test_input = torch.randn(1, 100, 1024, dtype=torch.float16, device="cuda")
    output = trt_transformer(test_input)
    print(f"  Input: {test_input.shape}")
    print(f"  Output: {output.shape}")

    # Benchmark
    print("\nBenchmarking...")
    for seq_len in [50, 100, 256, 512]:
        results = trt_transformer.benchmark(seq_len=seq_len)
        print(f"  Seq {seq_len:4d}: {results['mean_ms']:.2f} ms "
              f"({results['throughput_tokens_per_sec']:.0f} tok/s)")
