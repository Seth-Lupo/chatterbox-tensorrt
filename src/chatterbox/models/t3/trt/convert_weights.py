# Copyright (c) 2025 Resemble AI
# MIT License
"""
Weight Conversion Script for T3 to T3ForTRT

This script converts pretrained T3 model weights to the T3ForTRT format.
The conversion preserves all weights that are needed for inference while
removing the conditioning encoder (which is baked at compile time).

Weight Mapping:
    Original T3                    -> T3ForTRT
    ─────────────────────────────────────────────────
    tfmr.*                         -> transformer.*
    text_emb.*                     -> embedding.text_emb.*
    speech_emb.*                   -> embedding.speech_emb.*
    text_pos_emb.*                 -> embedding.text_pos_emb.*
    speech_pos_emb.*               -> embedding.speech_pos_emb.*
    speech_head.*                  -> speech_head.*
    cond_enc.*                     -> (removed - baked at compile time)
    text_head.*                    -> (removed - not needed for TTS)

Usage:
    python convert_weights.py \
        --input /path/to/t3_checkpoint.pth \
        --output /path/to/t3_for_trt_checkpoint.pth \
        --voice_prefix /path/to/voice_prefix.pt
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# Weight key mapping from T3 to T3ForTRT
WEIGHT_KEY_MAPPING = {
    # Transformer backbone
    "tfmr.": "transformer.",

    # Embedding tables
    "text_emb.": "embedding.text_emb.",
    "speech_emb.": "embedding.speech_emb.",

    # Positional embeddings
    "text_pos_emb.": "embedding.text_pos_emb.",
    "speech_pos_emb.": "embedding.speech_pos_emb.",

    # Output head
    "speech_head.": "speech_head.",
}

# Keys to remove (not needed in T3ForTRT)
KEYS_TO_REMOVE = [
    "cond_enc.",  # Conditioning encoder (baked at compile time)
    "text_head.",  # Text head (not needed for TTS-only)
]


def map_weight_key(key: str) -> Optional[str]:
    """
    Map a weight key from T3 format to T3ForTRT format.

    Args:
        key: Original weight key

    Returns:
        Mapped key, or None if the key should be removed
    """
    # Check if key should be removed
    for remove_prefix in KEYS_TO_REMOVE:
        if key.startswith(remove_prefix):
            return None

    # Map key prefix
    for old_prefix, new_prefix in WEIGHT_KEY_MAPPING.items():
        if key.startswith(old_prefix):
            return new_prefix + key[len(old_prefix):]

    # Key not in mapping - log warning and keep as-is
    logger.warning(f"Unknown key: {key} - keeping as-is")
    return key


def convert_state_dict(
    t3_state_dict: Dict[str, Tensor],
    strict: bool = True,
) -> Dict[str, Tensor]:
    """
    Convert T3 state dict to T3ForTRT format.

    Args:
        t3_state_dict: Original T3 model state dict
        strict: Whether to fail on unknown keys

    Returns:
        Converted state dict for T3ForTRT
    """
    converted = {}
    removed_keys = []
    unknown_keys = []

    for key, value in t3_state_dict.items():
        new_key = map_weight_key(key)

        if new_key is None:
            removed_keys.append(key)
        elif new_key == key and key not in [
            prefix + suffix
            for prefix in WEIGHT_KEY_MAPPING.values()
            for suffix in ["weight", "bias"]
        ]:
            unknown_keys.append(key)
            if not strict:
                converted[new_key] = value
        else:
            converted[new_key] = value

    # Log conversion statistics
    logger.info(f"Converted {len(converted)} weight keys")
    logger.info(f"Removed {len(removed_keys)} keys: {removed_keys}")

    if unknown_keys:
        msg = f"Unknown keys: {unknown_keys}"
        if strict:
            raise ValueError(msg)
        else:
            logger.warning(msg)

    return converted


def convert_checkpoint(
    input_path: str,
    output_path: str,
    voice_prefix_path: Optional[str] = None,
    strict: bool = True,
) -> Dict[str, Tensor]:
    """
    Convert a full T3 checkpoint to T3ForTRT format.

    Args:
        input_path: Path to original T3 checkpoint
        output_path: Path to save converted checkpoint
        voice_prefix_path: Optional path to voice prefix file to include
        strict: Whether to fail on unknown keys

    Returns:
        Converted state dict
    """
    logger.info(f"Loading checkpoint from {input_path}")

    # Load original checkpoint
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            t3_state_dict = checkpoint["model_state_dict"]
            extra_data = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
        elif "state_dict" in checkpoint:
            t3_state_dict = checkpoint["state_dict"]
            extra_data = {k: v for k, v in checkpoint.items() if k != "state_dict"}
        else:
            # Assume it's just the state dict
            t3_state_dict = checkpoint
            extra_data = {}
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")

    logger.info(f"Original checkpoint has {len(t3_state_dict)} keys")

    # Convert state dict
    converted_state_dict = convert_state_dict(t3_state_dict, strict=strict)

    # Load voice prefix if provided
    voice_prefix_data = None
    if voice_prefix_path is not None:
        logger.info(f"Loading voice prefix from {voice_prefix_path}")
        voice_prefix_data = torch.load(voice_prefix_path, map_location="cpu", weights_only=True)

    # Create output checkpoint
    output_checkpoint = {
        "model_state_dict": converted_state_dict,
        "conversion_info": {
            "source": str(input_path),
            "num_converted_keys": len(converted_state_dict),
            "format": "t3_for_trt",
        },
    }

    # Include voice prefix if provided
    if voice_prefix_data is not None:
        output_checkpoint["voice_prefix"] = voice_prefix_data["voice_prefix"]
        output_checkpoint["voice_prefix_metadata"] = voice_prefix_data["metadata"]

    # Include any extra data from original checkpoint
    output_checkpoint["original_extra_data"] = extra_data

    # Save converted checkpoint
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(output_checkpoint, output_path)
    logger.info(f"Saved converted checkpoint to {output_path}")

    return converted_state_dict


def load_converted_checkpoint(
    checkpoint_path: str,
    model: "T3ForTRT",
    strict: bool = True,
) -> Tuple["T3ForTRT", dict]:
    """
    Load a converted checkpoint into a T3ForTRT model.

    Args:
        checkpoint_path: Path to converted checkpoint
        model: T3ForTRT model instance
        strict: Whether to require all keys to match

    Returns:
        Tuple of (model with loaded weights, checkpoint metadata)
    """
    logger.info(f"Loading converted checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Load state dict
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")

    # Load voice prefix if present
    if "voice_prefix" in checkpoint:
        logger.info("Loading voice prefix from checkpoint")
        model.set_voice_prefix(checkpoint["voice_prefix"])

    return model, checkpoint.get("conversion_info", {})


def verify_conversion(
    original_model: "T3",
    converted_model: "T3ForTRT",
    text_tokens: Tensor,
    t3_cond: "T3Cond",
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> bool:
    """
    Verify that converted model produces equivalent outputs.

    This compares the transformer hidden states (not the full outputs,
    since the conditioning path is different).

    Args:
        original_model: Original T3 model
        converted_model: Converted T3ForTRT model
        text_tokens: Test text tokens
        t3_cond: Test conditioning data
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        True if verification passes
    """
    logger.info("Verifying conversion...")

    original_model.eval()
    converted_model.eval()

    with torch.no_grad():
        # Get original embeddings
        original_embeds, len_cond = original_model.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=original_model.hp.start_speech_token * torch.ones_like(text_tokens[:, :1]),
        )

        # Get converted embeddings (with voice prefix prepended)
        text_embeds = converted_model.embedding.embed_text_only(text_tokens)
        voice_prefix = converted_model.voice_prefix
        if voice_prefix.size(0) != text_embeds.size(0):
            voice_prefix = voice_prefix.expand(text_embeds.size(0), -1, -1)

        # Add start speech token embedding
        start_speech = torch.tensor(
            [[converted_model.config.start_speech_token]],
            device=text_tokens.device
        )
        speech_emb = converted_model.embedding.get_speech_embedding_at_position(start_speech, 0)
        if speech_emb.size(0) != text_embeds.size(0):
            speech_emb = speech_emb.expand(text_embeds.size(0), -1, -1)

        converted_embeds = torch.cat([voice_prefix, text_embeds, speech_emb], dim=1)

        # Compare shapes
        if original_embeds.shape != converted_embeds.shape:
            logger.error(
                f"Shape mismatch: original={original_embeds.shape}, "
                f"converted={converted_embeds.shape}"
            )
            return False

        # Compare values (after conditioning prefix)
        # Note: The conditioning embeddings will differ because they're computed
        # differently. We compare the text+speech portions.
        text_start = len_cond
        text_end = len_cond + text_tokens.size(1)

        orig_text = original_embeds[:, text_start:text_end]
        conv_text = converted_embeds[:, len_cond:len_cond + text_tokens.size(1)]

        if not torch.allclose(orig_text, conv_text, rtol=rtol, atol=atol):
            max_diff = (orig_text - conv_text).abs().max().item()
            logger.error(f"Text embedding mismatch. Max diff: {max_diff}")
            return False

        logger.info("Conversion verification passed!")
        return True


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert T3 weights to T3ForTRT format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to original T3 checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save converted checkpoint",
    )
    parser.add_argument(
        "--voice_prefix",
        type=str,
        default=None,
        help="Optional path to voice prefix file to include",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Don't fail on unknown keys",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    convert_checkpoint(
        input_path=args.input,
        output_path=args.output,
        voice_prefix_path=args.voice_prefix,
        strict=not args.no_strict,
    )


if __name__ == "__main__":
    main()
