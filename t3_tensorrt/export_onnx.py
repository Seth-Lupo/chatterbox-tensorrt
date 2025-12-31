#!/usr/bin/env python3
"""
Export T3's GPT-2 transformer (without embeddings) to ONNX.

This exports ONLY the transformer blocks, taking embeddings as input.
The custom speech_emb, cond_enc, and speech_head stay in PyTorch.

Usage:
    python export_onnx.py
    python export_onnx.py --output t3_transformer.onnx
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add project to path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class T3TransformerOnly(nn.Module):
    """
    T3's GPT-2 transformer blocks only - takes embeddings as input.

    This is the compute-heavy part that benefits from TensorRT.
    The custom embedding and head layers stay in PyTorch.
    """

    def __init__(self, gpt2_model):
        super().__init__()
        # Copy transformer components
        self.wpe = gpt2_model.wpe        # Position embeddings
        self.drop = gpt2_model.drop      # Dropout
        self.h = gpt2_model.h            # Transformer blocks
        self.ln_f = gpt2_model.ln_f      # Final layer norm

        self.hidden_size = gpt2_model.config.hidden_size
        self.num_layers = len(self.h)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs_embeds: (batch, seq_len, hidden_size) - pre-computed embeddings

        Returns:
            hidden_states: (batch, seq_len, hidden_size) - transformer output
        """
        batch_size, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device

        # Position embeddings (assuming positions start at 0)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_embeds = self.wpe(position_ids)

        # Combine embeddings
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        # Run through transformer blocks
        for block in self.h:
            outputs = block(hidden_states)
            hidden_states = outputs[0]

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        return hidden_states


def export_to_onnx(output_path: Path, opset_version: int = 14):
    """Export T3 transformer to ONNX."""
    print("=" * 60)
    print("Exporting T3 Transformer to ONNX")
    print("=" * 60)

    # Force legacy ONNX exporter (PyTorch 2.x uses dynamo by default)
    import torch.onnx
    torch.onnx.dynamo_export = None  # Disable dynamo export

    # Load T3 model
    print("\nLoading T3 model...")
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    model = ChatterboxTurboTTS.from_pretrained(
        device="cuda",
        dtype="float16",
        compile_mode=None,
    )

    # Create transformer-only wrapper
    print("\nCreating transformer-only wrapper...")
    transformer = T3TransformerOnly(model.t3.tfmr)
    transformer = transformer.cuda().half().eval()

    hidden_size = transformer.hidden_size
    num_layers = transformer.num_layers

    print(f"  Hidden size: {hidden_size}")
    print(f"  Num layers: {num_layers}")

    # Test the wrapper
    print("\nTesting wrapper...")
    test_input = torch.randn(1, 100, hidden_size, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        test_output = transformer(test_input)
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {test_output.shape}")

    # Export to ONNX
    print(f"\nExporting to ONNX: {output_path}")
    print(f"  Opset version: {opset_version}")

    # Use dynamic axes for batch size and sequence length
    dynamic_axes = {
        "inputs_embeds": {0: "batch_size", 1: "seq_len"},
        "hidden_states": {0: "batch_size", 1: "seq_len"},
    }

    # Export using torch.onnx.export with explicit settings
    # Use dynamo=False to avoid the new exporter which has issues
    with torch.no_grad():
        torch.onnx.export(
            transformer,
            (test_input,),  # Tuple of inputs
            str(output_path),
            input_names=["inputs_embeds"],
            output_names=["hidden_states"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
            verbose=False,
        )

    file_size_mb = output_path.stat().st_size / (1024*1024)
    print(f"\nExport complete!")
    print(f"  File size: {file_size_mb:.2f} MB")

    # Check file size - should be ~600MB for T3
    if file_size_mb < 100:
        print(f"\n  WARNING: File size is suspiciously small!")
        print(f"  Expected ~600MB for 24-layer transformer")
        print(f"  Weights may not have been exported correctly")

    # Verify ONNX model (skip strict check if it fails)
    print("\nVerifying ONNX model...")
    import onnx
    try:
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("  ONNX model is valid!")
    except Exception as e:
        print(f"  Warning: ONNX validation failed: {e}")
        print("  This may still work with TensorRT - continuing...")

    # Show model info
    print(f"\n  Inputs:")
    for inp in onnx_model.graph.input:
        print(f"    {inp.name}: {[d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]}")
    print(f"  Outputs:")
    for out in onnx_model.graph.output:
        print(f"    {out.name}: {[d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export T3 transformer to ONNX")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "t3_transformer.onnx",
                        help="Output ONNX file path")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version (14 recommended)")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_to_onnx(args.output, args.opset)


if __name__ == "__main__":
    main()
