# Copyright (c) 2025 Resemble AI
# MIT License
"""
T3 TensorRT-LLM Export Module

This module provides utilities for exporting Chatterbox Turbo (T3) models
to TensorRT-LLM format for optimized inference.

Key Components:
    - T3ForTRT: TensorRT-LLM compatible model class
    - T3ForTRTConfig: Configuration for the TRT-compatible model
    - extract_conditioning: Extract and save voice prefix embeddings
    - convert_weights: Convert T3 weights to T3ForTRT format
    - export_trtllm: Export to TensorRT-LLM checkpoint and build engine

Quick Start:
    # 1. Extract voice conditioning for a specific voice
    from chatterbox.models.t3.trt import extract_and_save_voice_prefix

    extract_and_save_voice_prefix(
        t3_model=original_model,
        speaker_emb=speaker_embedding,
        output_path="voice_prefix.pt",
        cond_prompt_speech_tokens=voice_prompt_tokens,
    )

    # 2. Convert model weights
    from chatterbox.models.t3.trt import convert_checkpoint

    convert_checkpoint(
        input_path="t3_checkpoint.pth",
        output_path="t3_for_trt_checkpoint.pth",
        voice_prefix_path="voice_prefix.pt",
    )

    # 3. Create T3ForTRT model
    from chatterbox.models.t3.trt import create_t3_for_trt, load_converted_checkpoint

    model = create_t3_for_trt(voice_prefix_path="voice_prefix.pt")
    model, _ = load_converted_checkpoint("t3_for_trt_checkpoint.pth", model)

    # 4. Export to TensorRT-LLM
    from chatterbox.models.t3.trt import full_export_pipeline

    full_export_pipeline(
        t3_checkpoint_path="t3_checkpoint.pth",
        voice_prefix_path="voice_prefix.pt",
        output_dir="trt_export/",
    )

For complete documentation, see the README.md in this module directory.
"""

# Model classes and configs
from .t3_for_trt import (
    T3ForTRT,
    T3ForTRTConfig,
    UnifiedEmbedding,
    create_t3_for_trt,
)

# Conditioning extraction
from .extract_conditioning import (
    extract_conditioning_embeddings,
    extract_and_save_voice_prefix,
    save_voice_prefix,
    load_voice_prefix,
    validate_voice_prefix,
)

# Weight conversion
from .convert_weights import (
    convert_state_dict,
    convert_checkpoint,
    load_converted_checkpoint,
    verify_conversion,
)

# TensorRT-LLM export
from .export_trtllm import (
    TRTLLMExportConfig,
    export_checkpoint,
    create_prompt_table,
    create_unified_embedding_table,
    convert_to_trtllm_weights,
    generate_build_script,
    save_build_script,
    full_export_pipeline,
)

# Validation
from .validate import (
    ValidationResult,
    test_embedding_equivalence,
    test_voice_prefix_shape,
    test_generation_equivalence,
    test_export_integrity,
    test_prompt_table,
    run_all_tests,
    print_validation_report,
)

__all__ = [
    # Model
    "T3ForTRT",
    "T3ForTRTConfig",
    "UnifiedEmbedding",
    "create_t3_for_trt",
    # Conditioning
    "extract_conditioning_embeddings",
    "extract_and_save_voice_prefix",
    "save_voice_prefix",
    "load_voice_prefix",
    "validate_voice_prefix",
    # Weights
    "convert_state_dict",
    "convert_checkpoint",
    "load_converted_checkpoint",
    "verify_conversion",
    # Export
    "TRTLLMExportConfig",
    "export_checkpoint",
    "create_prompt_table",
    "create_unified_embedding_table",
    "convert_to_trtllm_weights",
    "generate_build_script",
    "save_build_script",
    "full_export_pipeline",
    # Validation
    "ValidationResult",
    "test_embedding_equivalence",
    "test_voice_prefix_shape",
    "test_generation_equivalence",
    "test_export_integrity",
    "test_prompt_table",
    "run_all_tests",
    "print_validation_report",
]
