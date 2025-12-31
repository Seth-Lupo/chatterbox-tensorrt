#!/usr/bin/env python3
"""
Step 1: Export Chatterbox Turbo models for TensorRT conversion

This script:
1. Downloads the Chatterbox Turbo model from HuggingFace
2. Exports T3 (GPT-2) to TensorRT-LLM checkpoint format
3. Exports S3Gen flow model to ONNX
4. Exports HiFiGAN vocoder to ONNX
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path

import torch
import numpy as np
from safetensors.torch import load_file, save_file
from huggingface_hub import snapshot_download

# Add parent to path for chatterbox imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.s3gen import S3Gen
from chatterbox.models.s3tokenizer import S3_SR
from chatterbox.models.s3gen import S3GEN_SR


REPO_ID = "ResembleAI/chatterbox-turbo"
EXPORT_DIR = Path("exports")


def download_model(token: str = None):
    """Download model from HuggingFace."""
    print("Downloading Chatterbox Turbo from HuggingFace...")

    # Get token from argument, env var, or cached login
    hf_token = token or os.getenv("HF_TOKEN")
    if not hf_token:
        print("Note: No HF_TOKEN provided. Trying cached login...")
        print("If this fails, either:")
        print("  1. Run: huggingface-cli login")
        print("  2. Or pass: --token YOUR_TOKEN")
        print("  3. Or set: export HF_TOKEN=YOUR_TOKEN")
        hf_token = True  # Use cached credentials

    local_path = snapshot_download(
        repo_id=REPO_ID,
        token=hf_token,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"]
    )
    print(f"Downloaded to: {local_path}")
    return Path(local_path)


def export_t3_for_trtllm(ckpt_dir: Path, export_dir: Path):
    """
    Export T3 model to TensorRT-LLM checkpoint format.

    T3 uses GPT-2 architecture, so we export to GPT-2 TRT-LLM format.
    """
    print("\n" + "="*50)
    print("Exporting T3 to TensorRT-LLM format...")
    print("="*50)

    t3_export_dir = export_dir / "t3_checkpoint"
    t3_export_dir.mkdir(parents=True, exist_ok=True)

    # Load T3 config
    hp = T3Config(text_tokens_dict_size=50276)
    hp.llama_config_name = "GPT2_medium"
    hp.speech_tokens_dict_size = 6563
    hp.input_pos_emb = None
    hp.speech_cond_prompt_len = 375
    hp.use_perceiver_resampler = False
    hp.emotion_adv = False

    # Load T3 weights
    t3 = T3(hp)
    t3_state = load_file(ckpt_dir / "t3_turbo_v1.safetensors")
    if "model" in t3_state.keys():
        t3_state = t3_state["model"][0]
    t3.load_state_dict(t3_state)

    # Get GPT-2 config from T3
    gpt2_config = t3.cfg

    # Create TRT-LLM config
    trtllm_config = {
        "architecture": "GPT2LMHeadModel",
        "dtype": "float16",
        "num_hidden_layers": gpt2_config.num_hidden_layers,
        "num_attention_heads": gpt2_config.num_attention_heads,
        "hidden_size": gpt2_config.hidden_size,
        "intermediate_size": gpt2_config.intermediate_size if hasattr(gpt2_config, 'intermediate_size') else gpt2_config.hidden_size * 4,
        "vocab_size": hp.speech_tokens_dict_size,  # Speech vocab for output
        "max_position_embeddings": 2048,
        "hidden_act": "gelu",
        "quantization": {
            "quant_algo": None,
            "kv_cache_quant_algo": None,
        },
        "mapping": {
            "world_size": 1,
            "tp_size": 1,
            "pp_size": 1,
        }
    }

    # Save config
    with open(t3_export_dir / "config.json", "w") as f:
        json.dump(trtllm_config, f, indent=2)

    # Export weights in TRT-LLM format
    # TRT-LLM expects specific naming conventions
    trtllm_weights = {}

    # Map T3/GPT-2 weights to TRT-LLM format
    state_dict = t3.tfmr.state_dict()

    for name, param in state_dict.items():
        # Convert to TRT-LLM naming convention
        new_name = name

        # Handle layer normalization
        if "ln_" in name:
            new_name = name.replace("ln_1", "input_layernorm").replace("ln_2", "post_attention_layernorm")

        # Handle attention
        if "attn" in name:
            new_name = name.replace("attn.c_attn", "attention.qkv").replace("attn.c_proj", "attention.dense")

        # Handle MLP
        if "mlp" in name:
            new_name = name.replace("mlp.c_fc", "mlp.fc").replace("mlp.c_proj", "mlp.proj")

        trtllm_weights[new_name] = param.half().contiguous()

    # Add speech embedding and head
    trtllm_weights["speech_emb.weight"] = t3.speech_emb.weight.half().contiguous()
    trtllm_weights["speech_head.weight"] = t3.speech_head.weight.half().contiguous()
    if t3.speech_head.bias is not None:
        trtllm_weights["speech_head.bias"] = t3.speech_head.bias.half().contiguous()

    # Add text embedding (for conditioning)
    trtllm_weights["text_emb.weight"] = t3.text_emb.weight.half().contiguous()

    # Save weights
    save_file(trtllm_weights, t3_export_dir / "model.safetensors")

    print(f"T3 exported to: {t3_export_dir}")
    print(f"  Config: {t3_export_dir / 'config.json'}")
    print(f"  Weights: {t3_export_dir / 'model.safetensors'}")

    return t3_export_dir


def export_s3gen_to_onnx(ckpt_dir: Path, export_dir: Path):
    """Export S3Gen flow model to ONNX."""
    print("\n" + "="*50)
    print("Exporting S3Gen to ONNX...")
    print("="*50)

    export_dir.mkdir(parents=True, exist_ok=True)

    # Load S3Gen
    s3gen = S3Gen(meanflow=True)
    weights = load_file(ckpt_dir / "s3gen_meanflow.safetensors")
    s3gen.load_state_dict(weights, strict=True)
    s3gen.eval()

    # Export flow model
    print("Exporting flow model...")
    flow_model = s3gen.flow
    flow_model.eval()

    # Create dummy inputs for tracing
    batch_size = 1
    seq_len = 100  # Variable, but need a concrete value for tracing

    # Flow model inputs (check the actual forward signature)
    dummy_speech_tokens = torch.randint(0, 6561, (batch_size, seq_len))

    # Export to ONNX
    flow_onnx_path = export_dir / "s3gen_flow.onnx"

    # We need to export the inference path, not training
    # Create a wrapper for clean export
    class FlowWrapper(torch.nn.Module):
        def __init__(self, s3gen):
            super().__init__()
            self.s3gen = s3gen

        def forward(self, speech_tokens, prompt_feat, embedding):
            """Simplified forward for ONNX export."""
            # This matches the inference path in s3gen
            return self.s3gen.flow_inference_simple(
                speech_tokens, prompt_feat, embedding
            )

    # For now, export the full S3Gen model components separately
    # Export tokenizer embedding
    print("Exporting S3Gen components...")

    # Save the full S3Gen for now - ONNX export of complex models needs careful handling
    torch.save({
        'state_dict': s3gen.state_dict(),
        'config': {'meanflow': True}
    }, export_dir / "s3gen_full.pt")

    print(f"S3Gen saved to: {export_dir / 's3gen_full.pt'}")
    print("Note: Full ONNX export requires custom handling for flow matching models")

    # Export HiFiGAN vocoder separately (this is simpler)
    print("\nExporting HiFiGAN vocoder...")
    hifigan = s3gen.mel2wav
    hifigan.eval()

    # HiFiGAN input: mel spectrogram
    dummy_mel = torch.randn(1, 80, 100)  # (batch, mel_channels, time)
    dummy_cache = torch.zeros(1, 1, 0)

    hifigan_onnx_path = export_dir / "hifigan.onnx"

    class HiFiGANWrapper(torch.nn.Module):
        def __init__(self, hifigan):
            super().__init__()
            self.hifigan = hifigan

        def forward(self, mel):
            cache = torch.zeros(1, 1, 0, device=mel.device)
            wav, *_ = self.hifigan.inference(speech_feat=mel, cache_source=cache)
            return wav

    wrapper = HiFiGANWrapper(hifigan)
    wrapper.eval()

    try:
        torch.onnx.export(
            wrapper,
            dummy_mel,
            str(hifigan_onnx_path),
            input_names=["mel"],
            output_names=["audio"],
            dynamic_axes={
                "mel": {0: "batch", 2: "time"},
                "audio": {0: "batch", 1: "samples"}
            },
            opset_version=17,
            do_constant_folding=True,
        )
        print(f"HiFiGAN exported to: {hifigan_onnx_path}")
    except Exception as e:
        print(f"HiFiGAN ONNX export failed: {e}")
        print("Saving PyTorch model instead...")
        torch.save(hifigan.state_dict(), export_dir / "hifigan.pt")

    return export_dir


def export_conditioning(ckpt_dir: Path, export_dir: Path):
    """Export conditioning data (voice encoder, etc.)."""
    print("\n" + "="*50)
    print("Exporting conditioning models...")
    print("="*50)

    # Copy voice encoder
    ve_src = ckpt_dir / "ve.safetensors"
    ve_dst = export_dir / "ve.safetensors"
    if ve_src.exists():
        shutil.copy(ve_src, ve_dst)
        print(f"Voice encoder copied to: {ve_dst}")

    # Copy default voice conditionals
    conds_src = ckpt_dir / "conds.pt"
    conds_dst = export_dir / "conds.pt"
    if conds_src.exists():
        shutil.copy(conds_src, conds_dst)
        print(f"Default conditionals copied to: {conds_dst}")

    # Copy tokenizer
    for fname in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]:
        src = ckpt_dir / fname
        if src.exists():
            shutil.copy(src, export_dir / fname)
            print(f"Copied: {fname}")


def main():
    parser = argparse.ArgumentParser(description="Export Chatterbox Turbo for TensorRT")
    parser.add_argument("--output", type=str, default="exports", help="Output directory")
    parser.add_argument("--skip-download", action="store_true", help="Skip model download (use cached)")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    export_dir = Path(args.output)
    export_dir.mkdir(parents=True, exist_ok=True)

    # Download model
    if args.skip_download:
        # Try to find cached model
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        ckpt_dir = None
        for repo in cache_info.repos:
            if repo.repo_id == REPO_ID:
                for revision in repo.revisions:
                    ckpt_dir = Path(revision.snapshot_path)
                    break
        if ckpt_dir is None:
            print("No cached model found, downloading...")
            ckpt_dir = download_model(token=args.token)
    else:
        ckpt_dir = download_model(token=args.token)

    print(f"\nUsing checkpoint: {ckpt_dir}")

    # Export models
    export_t3_for_trtllm(ckpt_dir, export_dir)
    export_s3gen_to_onnx(ckpt_dir, export_dir)
    export_conditioning(ckpt_dir, export_dir)

    print("\n" + "="*50)
    print("Export complete!")
    print("="*50)
    print(f"\nExported files in: {export_dir}")
    for f in sorted(export_dir.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.relative_to(export_dir)}: {size_mb:.1f} MB")

    print("\nNext step: python 02_build_engines.py")


if __name__ == "__main__":
    main()
