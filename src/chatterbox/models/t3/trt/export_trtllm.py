# Copyright (c) 2025 Resemble AI
# MIT License
"""
TensorRT-LLM Export Utilities for T3ForTRT

This module provides utilities to export T3ForTRT models to TensorRT-LLM format.
The export process involves:

1. Converting model weights to TensorRT-LLM checkpoint format
2. Setting up prompt table for baked voice conditioning
3. Building the TensorRT engine with appropriate configuration

Key TensorRT-LLM Concepts:
- Prompt Tuning: Fixed embeddings prepended to each sequence (our voice prefix)
- KV Cache: Managed by TensorRT-LLM for efficient autoregressive generation
- Paged Attention: Efficient memory management for long sequences

Architecture Mapping:
    T3ForTRT                       -> TensorRT-LLM
    ─────────────────────────────────────────────────
    voice_prefix                   -> prompt_table[0]
    embedding.text_emb             -> embedding.vocab_embedding (partial)
    embedding.speech_emb           -> embedding.vocab_embedding (partial)
    embedding.text_pos_emb         -> (handled by custom plugin or fused)
    embedding.speech_pos_emb       -> (handled by custom plugin or fused)
    transformer                    -> model (LLaMA architecture)
    speech_head                    -> lm_head

Usage:
    from chatterbox.models.t3.trt import export_trtllm

    # Export to TensorRT-LLM checkpoint
    export_trtllm.export_checkpoint(
        model=t3_for_trt,
        output_dir="/path/to/trt_checkpoint",
        dtype="float16",
    )

    # Build TensorRT engine
    export_trtllm.build_engine(
        checkpoint_dir="/path/to/trt_checkpoint",
        engine_dir="/path/to/trt_engine",
        max_input_len=2048,
        max_output_len=4096,
    )
"""

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TRTLLMExportConfig:
    """Configuration for TensorRT-LLM export."""

    # Model architecture
    architecture: str = "LlamaForCausalLM"
    dtype: str = "float16"  # float16, bfloat16, or float32

    # Vocabulary
    vocab_size: int = 8898  # text_vocab + speech_vocab (704 + 8194)
    hidden_size: int = 1024

    # Transformer config
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    intermediate_size: int = 4096
    hidden_act: str = "silu"

    # Sequence lengths
    max_position_embeddings: int = 131072
    max_input_len: int = 2048
    max_output_len: int = 4096
    max_batch_size: int = 1

    # Prompt tuning (voice conditioning)
    use_prompt_tuning: bool = True
    max_prompt_embedding_table_size: int = 64  # Max tokens in prompt table

    # Quantization (optional)
    use_int8_kv_cache: bool = False
    use_fp8: bool = False

    # Attention
    use_gpt_attention_plugin: bool = True
    use_paged_context_fmha: bool = True

    # Output head
    output_vocab_size: int = 8194  # speech_vocab only for generation


@dataclass
class PromptTableEntry:
    """Entry in the prompt table for TensorRT-LLM."""

    prompt_id: int
    prompt_embedding: Tensor  # Shape: (prompt_len, hidden_size)
    prompt_len: int

    def to_numpy(self, dtype=np.float16) -> np.ndarray:
        """Convert to numpy array for TensorRT-LLM."""
        return self.prompt_embedding.cpu().numpy().astype(dtype)


def create_prompt_table(
    voice_prefix: Tensor,
    max_prompt_len: int = 64,
) -> Dict[str, np.ndarray]:
    """
    Create the prompt table for TensorRT-LLM.

    The prompt table is used for prompt tuning, where fixed embeddings
    are prepended to the input sequence. In our case, this is the
    baked voice conditioning.

    Args:
        voice_prefix: Voice conditioning embeddings, shape (1, P, hidden_size)
        max_prompt_len: Maximum prompt length (must be >= P)

    Returns:
        Dictionary with prompt table arrays for TensorRT-LLM
    """
    if voice_prefix.dim() == 3:
        voice_prefix = voice_prefix.squeeze(0)  # (P, hidden_size)

    prompt_len, hidden_size = voice_prefix.shape

    if prompt_len > max_prompt_len:
        raise ValueError(
            f"Voice prefix length {prompt_len} exceeds max_prompt_len {max_prompt_len}"
        )

    # Pad to max_prompt_len if needed
    if prompt_len < max_prompt_len:
        padding = torch.zeros(
            max_prompt_len - prompt_len, hidden_size,
            dtype=voice_prefix.dtype, device=voice_prefix.device
        )
        padded_prefix = torch.cat([voice_prefix, padding], dim=0)
    else:
        padded_prefix = voice_prefix

    # Create prompt table with single entry (prompt_id=0)
    prompt_table = {
        "prompt_embedding_table": padded_prefix.cpu().numpy().astype(np.float16),
        "prompt_vocab_size": 1,  # One voice = one prompt
        "prompt_lengths": np.array([prompt_len], dtype=np.int32),
    }

    logger.info(f"Created prompt table:")
    logger.info(f"  Prompt length: {prompt_len}")
    logger.info(f"  Max prompt length: {max_prompt_len}")
    logger.info(f"  Hidden size: {hidden_size}")

    return prompt_table


def create_unified_embedding_table(
    text_emb_weight: Tensor,
    speech_emb_weight: Tensor,
) -> Tensor:
    """
    Create unified embedding table from separate text and speech embeddings.

    The unified table concatenates text and speech embeddings:
    [0, text_vocab_size) -> text embeddings
    [text_vocab_size, text_vocab_size + speech_vocab_size) -> speech embeddings

    Args:
        text_emb_weight: Text embedding weights, shape (text_vocab, hidden_size)
        speech_emb_weight: Speech embedding weights, shape (speech_vocab, hidden_size)

    Returns:
        Unified embedding table, shape (total_vocab, hidden_size)
    """
    unified = torch.cat([text_emb_weight, speech_emb_weight], dim=0)
    logger.info(f"Created unified embedding table: {unified.shape}")
    return unified


def convert_to_trtllm_weights(
    model: "T3ForTRT",
    dtype: str = "float16",
) -> Dict[str, np.ndarray]:
    """
    Convert T3ForTRT model weights to TensorRT-LLM format.

    This maps the PyTorch model weights to the format expected by
    TensorRT-LLM's checkpoint loading.

    Args:
        model: T3ForTRT model instance
        dtype: Target dtype for weights

    Returns:
        Dictionary of numpy arrays for TensorRT-LLM
    """
    dtype_map = {
        "float16": np.float16,
        "bfloat16": np.float16,  # TRT-LLM converts bf16 internally
        "float32": np.float32,
    }
    np_dtype = dtype_map[dtype]

    weights = {}

    # Unified embedding table
    unified_emb = create_unified_embedding_table(
        model.embedding.text_emb.weight.data,
        model.embedding.speech_emb.weight.data,
    )
    weights["transformer.vocab_embedding.weight"] = unified_emb.cpu().numpy().astype(np_dtype)

    # Positional embeddings (stored separately for custom handling)
    weights["transformer.text_position_embedding.weight"] = (
        model.embedding.text_pos_emb.emb.weight.data.cpu().numpy().astype(np_dtype)
    )
    weights["transformer.speech_position_embedding.weight"] = (
        model.embedding.speech_pos_emb.emb.weight.data.cpu().numpy().astype(np_dtype)
    )

    # Transformer layers
    for name, param in model.transformer.named_parameters():
        trt_name = f"transformer.{name}"
        weights[trt_name] = param.data.cpu().numpy().astype(np_dtype)

    # Output head (speech logits)
    weights["lm_head.weight"] = (
        model.speech_head.weight.data.cpu().numpy().astype(np_dtype)
    )
    if model.speech_head.bias is not None:
        weights["lm_head.bias"] = (
            model.speech_head.bias.data.cpu().numpy().astype(np_dtype)
        )

    logger.info(f"Converted {len(weights)} weight tensors to TRT-LLM format")

    return weights


def export_checkpoint(
    model: "T3ForTRT",
    output_dir: Union[str, Path],
    config: Optional[TRTLLMExportConfig] = None,
) -> Path:
    """
    Export T3ForTRT model to TensorRT-LLM checkpoint format.

    This creates a directory with:
    - config.json: Model configuration
    - rank0.safetensors: Model weights
    - prompt_table.npy: Voice conditioning prompt table

    Args:
        model: T3ForTRT model instance
        output_dir: Output directory for checkpoint
        config: Optional export configuration

    Returns:
        Path to output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create default config if not provided
    if config is None:
        config = TRTLLMExportConfig(
            vocab_size=model.config.total_vocab_size,
            hidden_size=model.config.hidden_size,
            output_vocab_size=model.config.speech_vocab_size,
            max_prompt_embedding_table_size=model.config.voice_prefix_len + 16,
        )

    # Export model config
    config_dict = asdict(config)
    config_dict["model_type"] = "t3_for_trt"
    config_dict["text_vocab_size"] = model.config.text_vocab_size
    config_dict["speech_vocab_size"] = model.config.speech_vocab_size
    config_dict["voice_prefix_len"] = model.config.voice_prefix_len

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Saved config to {config_path}")

    # Export weights
    weights = convert_to_trtllm_weights(model, dtype=config.dtype)

    # Save weights (using safetensors if available, otherwise numpy)
    try:
        from safetensors.numpy import save_file
        weights_path = output_dir / "rank0.safetensors"
        save_file(weights, str(weights_path))
    except ImportError:
        logger.warning("safetensors not available, saving as numpy files")
        weights_dir = output_dir / "weights"
        weights_dir.mkdir(exist_ok=True)
        for name, weight in weights.items():
            np.save(weights_dir / f"{name}.npy", weight)
        weights_path = weights_dir

    logger.info(f"Saved weights to {weights_path}")

    # Export prompt table (voice conditioning)
    if model._voice_prefix is not None:
        prompt_table = create_prompt_table(
            model.voice_prefix,
            max_prompt_len=config.max_prompt_embedding_table_size,
        )
        prompt_table_path = output_dir / "prompt_table.npz"
        np.savez(prompt_table_path, **prompt_table)
        logger.info(f"Saved prompt table to {prompt_table_path}")

    logger.info(f"Export complete: {output_dir}")
    return output_dir


def generate_build_script(
    checkpoint_dir: Union[str, Path],
    engine_dir: Union[str, Path],
    tp_size: int = 1,
    pp_size: int = 1,
) -> str:
    """
    Generate a shell script for building the TensorRT-LLM engine.

    Args:
        checkpoint_dir: Directory with exported checkpoint
        engine_dir: Output directory for TensorRT engine
        tp_size: Tensor parallel size
        pp_size: Pipeline parallel size

    Returns:
        Shell script content
    """
    checkpoint_dir = Path(checkpoint_dir)
    engine_dir = Path(engine_dir)

    # Load config
    with open(checkpoint_dir / "config.json") as f:
        config = json.load(f)

    script = f"""#!/bin/bash
# TensorRT-LLM Engine Build Script for T3ForTRT
# Generated automatically - review before running

set -e

CHECKPOINT_DIR="{checkpoint_dir.absolute()}"
ENGINE_DIR="{engine_dir.absolute()}"
TP_SIZE={tp_size}
PP_SIZE={pp_size}

# Model parameters
MAX_INPUT_LEN={config.get('max_input_len', 2048)}
MAX_OUTPUT_LEN={config.get('max_output_len', 4096)}
MAX_BATCH_SIZE={config.get('max_batch_size', 1)}
DTYPE={config.get('dtype', 'float16')}

echo "Building TensorRT-LLM engine for T3ForTRT..."
echo "  Checkpoint: $CHECKPOINT_DIR"
echo "  Engine: $ENGINE_DIR"
echo "  TP: $TP_SIZE, PP: $PP_SIZE"

# Build engine using TensorRT-LLM
trtllm-build \\
    --checkpoint_dir "$CHECKPOINT_DIR" \\
    --output_dir "$ENGINE_DIR" \\
    --gemm_plugin $DTYPE \\
    --gpt_attention_plugin $DTYPE \\
    --max_input_len $MAX_INPUT_LEN \\
    --max_seq_len $(($MAX_INPUT_LEN + $MAX_OUTPUT_LEN)) \\
    --max_batch_size $MAX_BATCH_SIZE \\
    --max_num_tokens $(($MAX_BATCH_SIZE * $MAX_INPUT_LEN)) \\
    --use_paged_context_fmha enable \\
    --use_prompt_tuning enable \\
    --max_prompt_embedding_table_size {config.get('max_prompt_embedding_table_size', 64)} \\
    --tp_size $TP_SIZE \\
    --pp_size $PP_SIZE

echo "Engine build complete!"
echo "Engine saved to: $ENGINE_DIR"
"""

    return script


def save_build_script(
    checkpoint_dir: Union[str, Path],
    engine_dir: Union[str, Path],
    script_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Save the engine build script to a file.

    Args:
        checkpoint_dir: Directory with exported checkpoint
        engine_dir: Output directory for TensorRT engine
        script_path: Optional path for script (default: checkpoint_dir/build_engine.sh)

    Returns:
        Path to saved script
    """
    checkpoint_dir = Path(checkpoint_dir)

    if script_path is None:
        script_path = checkpoint_dir / "build_engine.sh"
    else:
        script_path = Path(script_path)

    script = generate_build_script(checkpoint_dir, engine_dir)

    with open(script_path, "w") as f:
        f.write(script)

    # Make executable
    script_path.chmod(script_path.stat().st_mode | 0o111)

    logger.info(f"Saved build script to {script_path}")
    return script_path


# =============================================================================
# High-Level Export Pipeline
# =============================================================================

def full_export_pipeline(
    t3_checkpoint_path: str,
    voice_prefix_path: str,
    output_dir: str,
    engine_dir: Optional[str] = None,
    multilingual: bool = False,
) -> Dict[str, Path]:
    """
    Run the full export pipeline from T3 checkpoint to TensorRT-LLM.

    This is a convenience function that:
    1. Loads the T3 checkpoint
    2. Converts to T3ForTRT format
    3. Loads the voice prefix
    4. Exports to TensorRT-LLM checkpoint format
    5. Generates the engine build script

    Args:
        t3_checkpoint_path: Path to original T3 checkpoint
        voice_prefix_path: Path to extracted voice prefix
        output_dir: Output directory for TRT-LLM checkpoint
        engine_dir: Output directory for TRT engine (default: output_dir/engine)
        multilingual: Whether to use multilingual configuration

    Returns:
        Dictionary with paths to generated files
    """
    from ..modules.t3_config import T3Config
    from .t3_for_trt import T3ForTRT, T3ForTRTConfig
    from .convert_weights import convert_state_dict
    from .extract_conditioning import load_voice_prefix, validate_voice_prefix

    output_dir = Path(output_dir)
    if engine_dir is None:
        engine_dir = output_dir / "engine"
    else:
        engine_dir = Path(engine_dir)

    logger.info("=" * 60)
    logger.info("T3 to TensorRT-LLM Export Pipeline")
    logger.info("=" * 60)

    # Step 1: Load and convert T3 checkpoint
    logger.info("\n[1/4] Loading T3 checkpoint...")
    t3_config = T3Config.multilingual() if multilingual else T3Config.english_only()
    checkpoint = torch.load(t3_checkpoint_path, map_location="cpu", weights_only=False)

    if "model_state_dict" in checkpoint:
        t3_state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        t3_state_dict = checkpoint["state_dict"]
    else:
        t3_state_dict = checkpoint

    converted_state_dict = convert_state_dict(t3_state_dict, strict=False)

    # Step 2: Load voice prefix
    logger.info("\n[2/4] Loading voice prefix...")
    voice_prefix, metadata = load_voice_prefix(voice_prefix_path)
    validate_voice_prefix(voice_prefix, metadata)
    voice_prefix_len = metadata["len_cond"]

    # Step 3: Create T3ForTRT model and load weights
    logger.info("\n[3/4] Creating T3ForTRT model...")
    config = T3ForTRTConfig.from_t3_config(t3_config, voice_prefix_len=voice_prefix_len)
    model = T3ForTRT(config)

    # Load converted weights
    missing, unexpected = model.load_state_dict(converted_state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys (expected for fresh T3ForTRT): {len(missing)}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")

    # Set voice prefix
    model.set_voice_prefix(voice_prefix)

    # Step 4: Export to TensorRT-LLM format
    logger.info("\n[4/4] Exporting to TensorRT-LLM format...")
    checkpoint_dir = export_checkpoint(model, output_dir)

    # Generate build script
    build_script_path = save_build_script(checkpoint_dir, engine_dir)

    logger.info("\n" + "=" * 60)
    logger.info("Export Complete!")
    logger.info("=" * 60)
    logger.info(f"\nGenerated files:")
    logger.info(f"  Checkpoint: {checkpoint_dir}")
    logger.info(f"  Build script: {build_script_path}")
    logger.info(f"\nTo build the TensorRT engine, run:")
    logger.info(f"  bash {build_script_path}")

    return {
        "checkpoint_dir": checkpoint_dir,
        "build_script": build_script_path,
        "engine_dir": engine_dir,
    }


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Export T3ForTRT model to TensorRT-LLM format"
    )
    parser.add_argument(
        "--t3_checkpoint",
        type=str,
        required=True,
        help="Path to original T3 checkpoint",
    )
    parser.add_argument(
        "--voice_prefix",
        type=str,
        required=True,
        help="Path to extracted voice prefix file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for TRT-LLM checkpoint",
    )
    parser.add_argument(
        "--engine_dir",
        type=str,
        default=None,
        help="Output directory for TRT engine (default: output_dir/engine)",
    )
    parser.add_argument(
        "--multilingual",
        action="store_true",
        help="Use multilingual model configuration",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    full_export_pipeline(
        t3_checkpoint_path=args.t3_checkpoint,
        voice_prefix_path=args.voice_prefix,
        output_dir=args.output_dir,
        engine_dir=args.engine_dir,
        multilingual=args.multilingual,
    )


if __name__ == "__main__":
    main()
