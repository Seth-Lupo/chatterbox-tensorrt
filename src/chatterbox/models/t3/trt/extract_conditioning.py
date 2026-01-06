# Copyright (c) 2025 Resemble AI
# MIT License
"""
Conditioning Extraction Script for TensorRT-LLM Export

This script extracts the voice conditioning embeddings from a trained T3 model
for a specific voice. The extracted embeddings are saved as a fixed-size tensor
that can be used as a prompt table in TensorRT-LLM.

The conditioning prefix consists of:
- Speaker embedding projection: (1, 1, dim)
- Perceiver-resampled voice prompt: (1, 32, dim)
- Emotion adversarial conditioning: (1, 1, dim)
Total: (1, 34, dim) for use_perceiver_resampler=True

Usage:
    python extract_conditioning.py \
        --checkpoint /path/to/t3_checkpoint.pth \
        --voice_audio /path/to/reference_voice.wav \
        --output /path/to/voice_prefix.pt \
        --emotion_adv 0.5
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def extract_conditioning_embeddings(
    t3_model: "T3",
    speaker_emb: Tensor,
    cond_prompt_speech_tokens: Optional[Tensor] = None,
    emotion_adv: float = 0.5,
) -> Tuple[Tensor, dict]:
    """
    Extract the conditioning embeddings from a T3 model for a specific voice.

    This captures the EXACT embeddings that would be concatenated at the start
    of the sequence in the original model's prepare_input_embeds method.

    Args:
        t3_model: Trained T3 model instance
        speaker_emb: Speaker embedding vector, shape (256,) or (1, 256)
        cond_prompt_speech_tokens: Optional voice prompt tokens, shape (1, seq_len)
        emotion_adv: Emotion adversarial value (default 0.5)

    Returns:
        cond_embeds: Conditioning embeddings, shape (1, len_cond, dim)
        metadata: Dictionary with extraction metadata
    """
    from ..modules.cond_enc import T3Cond

    device = t3_model.device
    dtype = next(t3_model.parameters()).dtype

    # Ensure speaker_emb is properly shaped
    speaker_emb = speaker_emb.to(device=device, dtype=dtype)
    if speaker_emb.dim() == 1:
        speaker_emb = speaker_emb.unsqueeze(0)

    # Prepare emotion adversarial tensor
    emotion_adv_tensor = torch.tensor([[emotion_adv]], device=device, dtype=dtype)

    # Prepare conditioning data structure
    t3_cond = T3Cond(
        speaker_emb=speaker_emb,
        clap_emb=None,
        cond_prompt_speech_tokens=cond_prompt_speech_tokens.to(device) if cond_prompt_speech_tokens is not None else None,
        cond_prompt_speech_emb=None,  # Will be computed by prepare_conditioning
        emotion_adv=emotion_adv_tensor,
    )

    # Extract conditioning embeddings using the model's own method
    with torch.no_grad():
        cond_embeds = t3_model.prepare_conditioning(t3_cond)

    # Compute metadata
    metadata = {
        "len_cond": cond_embeds.size(1),
        "hidden_size": cond_embeds.size(2),
        "dtype": str(cond_embeds.dtype),
        "use_perceiver_resampler": t3_model.hp.use_perceiver_resampler,
        "emotion_adv_value": emotion_adv,
        "has_voice_prompt": cond_prompt_speech_tokens is not None,
        "voice_prompt_len": cond_prompt_speech_tokens.size(1) if cond_prompt_speech_tokens is not None else 0,
    }

    logger.info(f"Extracted conditioning embeddings: shape={cond_embeds.shape}")
    logger.info(f"Metadata: {metadata}")

    return cond_embeds, metadata


def save_voice_prefix(
    cond_embeds: Tensor,
    metadata: dict,
    output_path: str,
    dtype: torch.dtype = torch.float16,
):
    """
    Save the extracted conditioning embeddings as a voice prefix file.

    The saved file contains:
    - 'voice_prefix': The conditioning embeddings in specified dtype
    - 'metadata': Extraction metadata for validation

    Args:
        cond_embeds: Conditioning embeddings tensor
        metadata: Extraction metadata dictionary
        output_path: Path to save the voice prefix file
        dtype: Data type for saving (default float16 for TensorRT)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to specified dtype
    voice_prefix = cond_embeds.to(dtype=dtype, device="cpu")

    save_dict = {
        "voice_prefix": voice_prefix,
        "metadata": metadata,
    }

    torch.save(save_dict, output_path)
    logger.info(f"Saved voice prefix to {output_path}")
    logger.info(f"  Shape: {voice_prefix.shape}")
    logger.info(f"  Dtype: {voice_prefix.dtype}")
    logger.info(f"  Size: {output_path.stat().st_size / 1024:.2f} KB")


def load_voice_prefix(prefix_path: str) -> Tuple[Tensor, dict]:
    """
    Load a saved voice prefix file.

    Args:
        prefix_path: Path to the voice prefix file

    Returns:
        voice_prefix: The conditioning embeddings tensor
        metadata: Extraction metadata dictionary
    """
    data = torch.load(prefix_path, map_location="cpu", weights_only=True)
    return data["voice_prefix"], data["metadata"]


def extract_and_save_voice_prefix(
    t3_model: "T3",
    speaker_emb: Tensor,
    output_path: str,
    cond_prompt_speech_tokens: Optional[Tensor] = None,
    emotion_adv: float = 0.5,
    dtype: torch.dtype = torch.float16,
) -> Tuple[Tensor, dict]:
    """
    Convenience function to extract and save voice prefix in one call.

    Args:
        t3_model: Trained T3 model instance
        speaker_emb: Speaker embedding vector
        output_path: Path to save the voice prefix file
        cond_prompt_speech_tokens: Optional voice prompt tokens
        emotion_adv: Emotion adversarial value
        dtype: Data type for saving

    Returns:
        voice_prefix: The extracted conditioning embeddings
        metadata: Extraction metadata dictionary
    """
    cond_embeds, metadata = extract_conditioning_embeddings(
        t3_model=t3_model,
        speaker_emb=speaker_emb,
        cond_prompt_speech_tokens=cond_prompt_speech_tokens,
        emotion_adv=emotion_adv,
    )

    save_voice_prefix(
        cond_embeds=cond_embeds,
        metadata=metadata,
        output_path=output_path,
        dtype=dtype,
    )

    return cond_embeds.to(dtype=dtype), metadata


def validate_voice_prefix(
    voice_prefix: Tensor,
    metadata: dict,
    expected_hidden_size: int = 1024,
) -> bool:
    """
    Validate a loaded voice prefix against expected parameters.

    Args:
        voice_prefix: The loaded conditioning embeddings
        metadata: The loaded metadata
        expected_hidden_size: Expected hidden dimension

    Returns:
        True if validation passes, raises ValueError otherwise
    """
    # Check shape
    if voice_prefix.dim() != 3:
        raise ValueError(f"Expected 3D tensor, got {voice_prefix.dim()}D")

    if voice_prefix.size(0) != 1:
        raise ValueError(f"Expected batch size 1, got {voice_prefix.size(0)}")

    if voice_prefix.size(2) != expected_hidden_size:
        raise ValueError(
            f"Expected hidden size {expected_hidden_size}, got {voice_prefix.size(2)}"
        )

    # Check metadata consistency
    if metadata["hidden_size"] != expected_hidden_size:
        raise ValueError(
            f"Metadata hidden_size {metadata['hidden_size']} doesn't match expected {expected_hidden_size}"
        )

    if metadata["len_cond"] != voice_prefix.size(1):
        raise ValueError(
            f"Metadata len_cond {metadata['len_cond']} doesn't match tensor shape {voice_prefix.size(1)}"
        )

    logger.info("Voice prefix validation passed")
    return True


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract voice conditioning embeddings from T3 model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to T3 model checkpoint",
    )
    parser.add_argument(
        "--speaker_emb",
        type=str,
        required=True,
        help="Path to speaker embedding file (.pt) or audio file for extraction",
    )
    parser.add_argument(
        "--voice_prompt_tokens",
        type=str,
        default=None,
        help="Path to voice prompt tokens file (.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for voice prefix file",
    )
    parser.add_argument(
        "--emotion_adv",
        type=float,
        default=0.5,
        help="Emotion adversarial value (default: 0.5)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "float32", "bfloat16"],
        default="float16",
        help="Data type for saved embeddings (default: float16)",
    )
    parser.add_argument(
        "--multilingual",
        action="store_true",
        help="Use multilingual model configuration",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Import here to avoid circular imports
    from ..t3 import T3
    from ..modules.t3_config import T3Config

    # Load model
    logger.info(f"Loading T3 model from {args.checkpoint}")
    hp = T3Config.multilingual() if args.multilingual else T3Config.english_only()
    model = T3(hp)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Load speaker embedding
    logger.info(f"Loading speaker embedding from {args.speaker_emb}")
    speaker_emb = torch.load(args.speaker_emb, map_location="cpu", weights_only=True)

    # Load voice prompt tokens if provided
    voice_prompt_tokens = None
    if args.voice_prompt_tokens:
        logger.info(f"Loading voice prompt tokens from {args.voice_prompt_tokens}")
        voice_prompt_tokens = torch.load(
            args.voice_prompt_tokens, map_location="cpu", weights_only=True
        )

    # Parse dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Extract and save
    extract_and_save_voice_prefix(
        t3_model=model,
        speaker_emb=speaker_emb,
        output_path=args.output,
        cond_prompt_speech_tokens=voice_prompt_tokens,
        emotion_adv=args.emotion_adv,
        dtype=dtype,
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
