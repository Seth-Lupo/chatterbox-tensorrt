# Copyright (c) 2025 Resemble AI
# MIT License
"""
Validation and Testing Utilities for T3ForTRT Export

This module provides utilities to validate the correctness of the T3 to
TensorRT-LLM export pipeline. It includes:

1. Embedding Equivalence Test: Verify that T3ForTRT produces the same
   embeddings as the original T3 model (after conditioning prefix).

2. Generation Comparison Test: Compare generated speech tokens between
   original T3 and T3ForTRT models.

3. Export Integrity Test: Verify that exported checkpoint contains all
   required weights and configurations.

4. Prompt Table Test: Verify that the prompt table is correctly formatted
   and contains valid voice conditioning.

Usage:
    from chatterbox.models.t3.trt import validate

    # Run all validation tests
    results = validate.run_all_tests(
        t3_model=original_model,
        t3_for_trt_model=converted_model,
        test_text_tokens=text_tokens,
        test_t3_cond=conditioning_data,
    )

    # Or run individual tests
    validate.test_embedding_equivalence(t3_model, t3_for_trt_model, text_tokens, t3_cond)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation test."""

    test_name: str
    passed: bool
    message: str
    details: Optional[Dict] = None


def test_embedding_equivalence(
    t3_model: "T3",
    t3_for_trt_model: "T3ForTRT",
    text_tokens: Tensor,
    t3_cond: "T3Cond",
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> ValidationResult:
    """
    Test that T3ForTRT produces equivalent text embeddings.

    This compares the text portion of the embeddings (after the conditioning
    prefix) between the original and converted models.

    Args:
        t3_model: Original T3 model
        t3_for_trt_model: Converted T3ForTRT model
        text_tokens: Test text tokens
        t3_cond: Test conditioning data
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        ValidationResult with test outcome
    """
    t3_model.eval()
    t3_for_trt_model.eval()

    with torch.no_grad():
        # Get original text embeddings
        cond_emb = t3_model.prepare_conditioning(t3_cond)
        text_emb_orig = t3_model.text_emb(text_tokens)
        if t3_model.hp.input_pos_emb == "learned":
            text_emb_orig = text_emb_orig + t3_model.text_pos_emb(text_tokens)

        # Get converted text embeddings
        text_emb_conv = t3_for_trt_model.embedding.embed_text_only(text_tokens)

        # Compare
        if text_emb_orig.shape != text_emb_conv.shape:
            return ValidationResult(
                test_name="embedding_equivalence",
                passed=False,
                message=f"Shape mismatch: orig={text_emb_orig.shape}, conv={text_emb_conv.shape}",
            )

        is_close = torch.allclose(text_emb_orig, text_emb_conv, rtol=rtol, atol=atol)
        max_diff = (text_emb_orig - text_emb_conv).abs().max().item()
        mean_diff = (text_emb_orig - text_emb_conv).abs().mean().item()

        if is_close:
            return ValidationResult(
                test_name="embedding_equivalence",
                passed=True,
                message="Text embeddings match within tolerance",
                details={"max_diff": max_diff, "mean_diff": mean_diff},
            )
        else:
            return ValidationResult(
                test_name="embedding_equivalence",
                passed=False,
                message=f"Text embeddings differ: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}",
                details={"max_diff": max_diff, "mean_diff": mean_diff},
            )


def test_voice_prefix_shape(
    t3_for_trt_model: "T3ForTRT",
    expected_len: Optional[int] = None,
) -> ValidationResult:
    """
    Test that voice prefix has correct shape and is set.

    Args:
        t3_for_trt_model: Converted T3ForTRT model
        expected_len: Optional expected prefix length

    Returns:
        ValidationResult with test outcome
    """
    try:
        voice_prefix = t3_for_trt_model.voice_prefix
    except RuntimeError as e:
        return ValidationResult(
            test_name="voice_prefix_shape",
            passed=False,
            message=f"Voice prefix not set: {e}",
        )

    # Check shape
    if voice_prefix.dim() != 3:
        return ValidationResult(
            test_name="voice_prefix_shape",
            passed=False,
            message=f"Expected 3D tensor, got {voice_prefix.dim()}D",
        )

    if voice_prefix.size(0) != 1:
        return ValidationResult(
            test_name="voice_prefix_shape",
            passed=False,
            message=f"Expected batch size 1, got {voice_prefix.size(0)}",
        )

    if voice_prefix.size(2) != t3_for_trt_model.config.hidden_size:
        return ValidationResult(
            test_name="voice_prefix_shape",
            passed=False,
            message=f"Hidden size mismatch: {voice_prefix.size(2)} vs {t3_for_trt_model.config.hidden_size}",
        )

    if expected_len is not None and voice_prefix.size(1) != expected_len:
        return ValidationResult(
            test_name="voice_prefix_shape",
            passed=False,
            message=f"Prefix length mismatch: {voice_prefix.size(1)} vs expected {expected_len}",
        )

    return ValidationResult(
        test_name="voice_prefix_shape",
        passed=True,
        message=f"Voice prefix shape OK: {voice_prefix.shape}",
        details={"shape": list(voice_prefix.shape)},
    )


def test_generation_equivalence(
    t3_model: "T3",
    t3_for_trt_model: "T3ForTRT",
    text_tokens: Tensor,
    t3_cond: "T3Cond",
    max_tokens: int = 10,
    seed: int = 42,
) -> ValidationResult:
    """
    Test that T3ForTRT produces similar generation behavior.

    Note: Due to different random states and potential numerical differences,
    we check for similar logit distributions rather than exact token matches.

    Args:
        t3_model: Original T3 model
        t3_for_trt_model: Converted T3ForTRT model
        text_tokens: Test text tokens
        t3_cond: Test conditioning data
        max_tokens: Number of tokens to generate
        seed: Random seed for reproducibility

    Returns:
        ValidationResult with test outcome
    """
    t3_model.eval()
    t3_for_trt_model.eval()

    with torch.no_grad():
        # Get initial logits from original model
        torch.manual_seed(seed)

        # Prepare original embeddings
        speech_start = t3_model.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        embeds_orig, len_cond = t3_model.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_start,
        )

        # Run through transformer
        tfmr_out_orig = t3_model.tfmr(
            inputs_embeds=embeds_orig,
            use_cache=False,
            return_dict=True,
        )
        logits_orig = t3_model.speech_head(tfmr_out_orig.last_hidden_state[:, -1:, :])

        # Get initial logits from converted model
        torch.manual_seed(seed)

        # Prepare converted embeddings
        text_embeds = t3_for_trt_model.embedding.embed_text_only(text_tokens)
        voice_prefix = t3_for_trt_model.voice_prefix.to(dtype=text_embeds.dtype)

        # Add start speech token
        start_speech = torch.tensor(
            [[t3_for_trt_model.config.start_speech_token]],
            device=text_tokens.device
        )
        speech_emb = t3_for_trt_model.embedding.get_speech_embedding_at_position(start_speech, 0)

        embeds_conv = torch.cat([voice_prefix, text_embeds, speech_emb], dim=1)

        # Run through transformer
        tfmr_out_conv = t3_for_trt_model.transformer(
            inputs_embeds=embeds_conv,
            use_cache=False,
            return_dict=True,
        )
        logits_conv = t3_for_trt_model.speech_head(tfmr_out_conv.last_hidden_state[:, -1:, :])

        # Compare logit distributions (top-k similarity)
        k = 10
        _, top_orig = torch.topk(logits_orig.squeeze(), k)
        _, top_conv = torch.topk(logits_conv.squeeze(), k)

        # Check overlap in top-k tokens
        overlap = len(set(top_orig.tolist()) & set(top_conv.tolist()))
        overlap_ratio = overlap / k

        # Also check cosine similarity of logit distributions
        cos_sim = torch.nn.functional.cosine_similarity(
            logits_orig.view(1, -1),
            logits_conv.view(1, -1),
        ).item()

        details = {
            "top_k_overlap": overlap,
            "overlap_ratio": overlap_ratio,
            "cosine_similarity": cos_sim,
            "top_orig": top_orig.tolist(),
            "top_conv": top_conv.tolist(),
        }

        # Pass if cosine similarity is high and overlap is reasonable
        if cos_sim > 0.9 and overlap_ratio >= 0.5:
            return ValidationResult(
                test_name="generation_equivalence",
                passed=True,
                message=f"Generation behavior similar: cos_sim={cos_sim:.4f}, overlap={overlap}/{k}",
                details=details,
            )
        else:
            return ValidationResult(
                test_name="generation_equivalence",
                passed=False,
                message=f"Generation behavior differs: cos_sim={cos_sim:.4f}, overlap={overlap}/{k}",
                details=details,
            )


def test_export_integrity(
    checkpoint_dir: str,
) -> ValidationResult:
    """
    Test that exported checkpoint contains all required files.

    Args:
        checkpoint_dir: Path to exported TensorRT-LLM checkpoint

    Returns:
        ValidationResult with test outcome
    """
    checkpoint_dir = Path(checkpoint_dir)

    required_files = ["config.json"]
    optional_files = ["rank0.safetensors", "prompt_table.npz", "build_engine.sh"]

    missing = []
    found = []

    for f in required_files:
        if (checkpoint_dir / f).exists():
            found.append(f)
        else:
            missing.append(f)

    for f in optional_files:
        if (checkpoint_dir / f).exists():
            found.append(f)

    if missing:
        return ValidationResult(
            test_name="export_integrity",
            passed=False,
            message=f"Missing required files: {missing}",
            details={"found": found, "missing": missing},
        )

    # Validate config.json
    import json
    try:
        with open(checkpoint_dir / "config.json") as f:
            config = json.load(f)

        required_keys = ["vocab_size", "hidden_size", "architecture"]
        missing_keys = [k for k in required_keys if k not in config]

        if missing_keys:
            return ValidationResult(
                test_name="export_integrity",
                passed=False,
                message=f"Config missing keys: {missing_keys}",
                details={"found_files": found, "config_keys": list(config.keys())},
            )

    except Exception as e:
        return ValidationResult(
            test_name="export_integrity",
            passed=False,
            message=f"Failed to parse config.json: {e}",
        )

    return ValidationResult(
        test_name="export_integrity",
        passed=True,
        message=f"Export integrity OK: {len(found)} files found",
        details={"found": found, "config": config},
    )


def test_prompt_table(
    prompt_table_path: str,
    expected_hidden_size: int = 1024,
) -> ValidationResult:
    """
    Test that prompt table is correctly formatted.

    Args:
        prompt_table_path: Path to prompt table file
        expected_hidden_size: Expected hidden dimension

    Returns:
        ValidationResult with test outcome
    """
    try:
        data = np.load(prompt_table_path)
    except Exception as e:
        return ValidationResult(
            test_name="prompt_table",
            passed=False,
            message=f"Failed to load prompt table: {e}",
        )

    required_keys = ["prompt_embedding_table", "prompt_vocab_size", "prompt_lengths"]
    missing_keys = [k for k in required_keys if k not in data]

    if missing_keys:
        return ValidationResult(
            test_name="prompt_table",
            passed=False,
            message=f"Missing keys in prompt table: {missing_keys}",
            details={"found_keys": list(data.keys())},
        )

    prompt_emb = data["prompt_embedding_table"]
    prompt_len = data["prompt_lengths"][0]

    # Check shape
    if prompt_emb.ndim != 2:
        return ValidationResult(
            test_name="prompt_table",
            passed=False,
            message=f"Expected 2D prompt embedding, got {prompt_emb.ndim}D",
        )

    if prompt_emb.shape[1] != expected_hidden_size:
        return ValidationResult(
            test_name="prompt_table",
            passed=False,
            message=f"Hidden size mismatch: {prompt_emb.shape[1]} vs {expected_hidden_size}",
        )

    # Check for valid values
    if np.any(np.isnan(prompt_emb)):
        return ValidationResult(
            test_name="prompt_table",
            passed=False,
            message="Prompt table contains NaN values",
        )

    if np.any(np.isinf(prompt_emb)):
        return ValidationResult(
            test_name="prompt_table",
            passed=False,
            message="Prompt table contains Inf values",
        )

    return ValidationResult(
        test_name="prompt_table",
        passed=True,
        message=f"Prompt table OK: shape={prompt_emb.shape}, len={prompt_len}",
        details={
            "shape": list(prompt_emb.shape),
            "prompt_len": int(prompt_len),
            "dtype": str(prompt_emb.dtype),
            "mean": float(np.mean(prompt_emb)),
            "std": float(np.std(prompt_emb)),
        },
    )


def run_all_tests(
    t3_model: Optional["T3"] = None,
    t3_for_trt_model: Optional["T3ForTRT"] = None,
    test_text_tokens: Optional[Tensor] = None,
    test_t3_cond: Optional["T3Cond"] = None,
    checkpoint_dir: Optional[str] = None,
    prompt_table_path: Optional[str] = None,
) -> List[ValidationResult]:
    """
    Run all applicable validation tests.

    Args:
        t3_model: Original T3 model (optional)
        t3_for_trt_model: Converted T3ForTRT model (optional)
        test_text_tokens: Test text tokens (optional)
        test_t3_cond: Test conditioning data (optional)
        checkpoint_dir: Path to exported checkpoint (optional)
        prompt_table_path: Path to prompt table (optional)

    Returns:
        List of ValidationResult objects
    """
    results = []

    # Model-based tests
    if t3_for_trt_model is not None:
        results.append(test_voice_prefix_shape(t3_for_trt_model))

    if all([t3_model, t3_for_trt_model, test_text_tokens, test_t3_cond]):
        results.append(test_embedding_equivalence(
            t3_model, t3_for_trt_model, test_text_tokens, test_t3_cond
        ))
        results.append(test_generation_equivalence(
            t3_model, t3_for_trt_model, test_text_tokens, test_t3_cond
        ))

    # File-based tests
    if checkpoint_dir is not None:
        results.append(test_export_integrity(checkpoint_dir))

    if prompt_table_path is not None:
        results.append(test_prompt_table(prompt_table_path))

    # Summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    logger.info(f"\n{'='*60}")
    logger.info(f"Validation Summary: {passed}/{total} tests passed")
    logger.info("=" * 60)

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        logger.info(f"[{status}] {result.test_name}: {result.message}")

    return results


def print_validation_report(results: List[ValidationResult]):
    """
    Print a detailed validation report.

    Args:
        results: List of ValidationResult objects
    """
    print("\n" + "=" * 70)
    print("T3 to TensorRT-LLM Export Validation Report")
    print("=" * 70 + "\n")

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    for i, result in enumerate(results, 1):
        status = "PASS" if result.passed else "FAIL"
        print(f"Test {i}: {result.test_name}")
        print(f"  Status: {status}")
        print(f"  Message: {result.message}")
        if result.details:
            print(f"  Details: {result.details}")
        print()

    print("-" * 70)
    print(f"Overall: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll tests passed! The export is valid.")
    else:
        print(f"\n{total - passed} test(s) failed. Please review and fix issues.")

    print("=" * 70)


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate T3 to TensorRT-LLM export"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Path to exported TRT-LLM checkpoint directory",
    )
    parser.add_argument(
        "--prompt_table",
        type=str,
        help="Path to prompt table file",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    results = run_all_tests(
        checkpoint_dir=args.checkpoint_dir,
        prompt_table_path=args.prompt_table,
    )

    print_validation_report(results)

    # Exit with error code if any tests failed
    if not all(r.passed for r in results):
        exit(1)


if __name__ == "__main__":
    main()
