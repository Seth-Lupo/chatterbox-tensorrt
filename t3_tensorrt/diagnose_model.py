#!/usr/bin/env python3
"""
Diagnose model issues - check weights, dtype, and compare configurations.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

output_dir = SCRIPT_DIR / "diagnose_output"
output_dir.mkdir(exist_ok=True)

text = "Hello, this is a test of the text to speech system."


def save_audio(audio, sr, filename):
    """Save audio tensor to wav file."""
    audio_cpu = audio.cpu().float()
    if audio_cpu.dim() == 2:
        audio_cpu = audio_cpu.squeeze(0)
    audio_np = audio_cpu.numpy()

    # Print stats
    print(f"  Audio stats: min={audio_np.min():.4f}, max={audio_np.max():.4f}, "
          f"mean={audio_np.mean():.4f}, std={audio_np.std():.4f}")

    # Normalize and save
    audio_norm = audio_np / (np.abs(audio_np).max() + 1e-8)
    audio_int16 = (audio_norm * 32767).astype(np.int16)

    path = output_dir / filename
    wavfile.write(str(path), sr, audio_int16)
    print(f"  Saved: {path}")
    return path


def check_model_weights(model, name="model"):
    """Check model weights for NaN, Inf, or suspicious values."""
    print(f"\n{'='*60}")
    print(f"Checking {name} weights")
    print(f"{'='*60}")

    issues = []
    for param_name, param in model.named_parameters():
        if torch.isnan(param).any():
            issues.append(f"  NaN in {param_name}")
        if torch.isinf(param).any():
            issues.append(f"  Inf in {param_name}")

        # Check for suspiciously large values
        max_val = param.abs().max().item()
        if max_val > 1000:
            issues.append(f"  Large values in {param_name}: max={max_val:.2f}")

    if issues:
        print("ISSUES FOUND:")
        for issue in issues[:20]:  # Limit output
            print(issue)
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more")
    else:
        print("No obvious weight issues found.")

    return len(issues) == 0


def test_float32():
    """Test with float32 instead of float16."""
    print("\n" + "=" * 60)
    print("TEST 1: Float32 (full precision)")
    print("=" * 60)

    from chatterbox.tts_turbo import ChatterboxTurboTTS

    print("Loading model with float32...")
    model = ChatterboxTurboTTS.from_pretrained(
        device="cuda",
        dtype="float32",  # Full precision
        compile_mode=None,
    )

    check_model_weights(model.t3, "T3")

    print(f"\nGenerating: \"{text}\"")
    audio = model.generate(text=text)
    save_audio(audio, model.sr, "test_float32.wav")

    del model
    torch.cuda.empty_cache()


def test_float16():
    """Test with float16."""
    print("\n" + "=" * 60)
    print("TEST 2: Float16 (half precision)")
    print("=" * 60)

    from chatterbox.tts_turbo import ChatterboxTurboTTS

    print("Loading model with float16...")
    model = ChatterboxTurboTTS.from_pretrained(
        device="cuda",
        dtype="float16",
        compile_mode=None,
    )

    check_model_weights(model.t3, "T3")

    print(f"\nGenerating: \"{text}\"")
    audio = model.generate(text=text)
    save_audio(audio, model.sr, "test_float16.wav")

    del model
    torch.cuda.empty_cache()


def test_cpu():
    """Test on CPU with float32."""
    print("\n" + "=" * 60)
    print("TEST 3: CPU with Float32")
    print("=" * 60)

    from chatterbox.tts_turbo import ChatterboxTurboTTS

    print("Loading model on CPU...")
    try:
        model = ChatterboxTurboTTS.from_pretrained(
            device="cpu",
            dtype="float32",
            compile_mode=None,
        )

        print(f"\nGenerating: \"{text}\"")
        audio = model.generate(text=text)
        save_audio(audio, model.sr, "test_cpu.wav")

        del model
    except Exception as e:
        print(f"CPU test failed: {e}")


def test_different_params():
    """Test with different generation parameters."""
    print("\n" + "=" * 60)
    print("TEST 4: Different generation parameters")
    print("=" * 60)

    from chatterbox.tts_turbo import ChatterboxTurboTTS

    print("Loading model...")
    model = ChatterboxTurboTTS.from_pretrained(
        device="cuda",
        dtype="float32",
        compile_mode=None,
    )

    param_sets = [
        {"temperature": 0.5, "top_p": 0.9, "top_k": 50, "name": "conservative"},
        {"temperature": 1.0, "top_p": 0.95, "top_k": 1000, "name": "default"},
        {"temperature": 0.1, "top_p": 0.5, "top_k": 10, "name": "very_conservative"},
    ]

    for params in param_sets:
        name = params.pop("name")
        print(f"\nGenerating with {name} params: {params}")
        audio = model.generate(text=text, **params)
        save_audio(audio, model.sr, f"test_params_{name}.wav")

    del model
    torch.cuda.empty_cache()


def check_intermediate_outputs():
    """Check intermediate outputs in the pipeline."""
    print("\n" + "=" * 60)
    print("TEST 5: Checking intermediate outputs")
    print("=" * 60)

    from chatterbox.tts_turbo import ChatterboxTurboTTS

    print("Loading model...")
    model = ChatterboxTurboTTS.from_pretrained(
        device="cuda",
        dtype="float32",
        compile_mode=None,
    )

    # Prepare conditionals
    model.prepare_conditionals(audio_prompt_path=None)

    # Get text tokens
    from chatterbox.models.t3.modules.normalizer import punc_norm
    text_normalized = punc_norm(text)
    text_tokens = model.tokenizer(text_normalized, return_tensors="pt", padding=True, truncation=True)
    text_tokens = text_tokens.input_ids.to(model.device)

    print(f"\nText tokens shape: {text_tokens.shape}")
    print(f"Text tokens: {text_tokens[0, :20].tolist()}...")  # First 20

    # Generate speech tokens
    print("\nGenerating speech tokens...")
    speech_tokens = model.t3.inference_turbo(
        t3_cond=model.conds.t3,
        text_tokens=text_tokens,
        temperature=0.8,
        top_k=1000,
        top_p=0.95,
        repetition_penalty=1.2,
    )

    print(f"Speech tokens shape: {speech_tokens.shape}")
    print(f"Speech tokens range: {speech_tokens.min().item()} to {speech_tokens.max().item()}")
    print(f"Speech tokens sample: {speech_tokens[:20].tolist()}...")

    # Check for unusual patterns
    unique_tokens = torch.unique(speech_tokens)
    print(f"Unique tokens: {len(unique_tokens)}")

    # Check token distribution
    token_counts = torch.bincount(speech_tokens.clamp(0, 6560), minlength=6561)
    top_tokens = torch.topk(token_counts, 10)
    print(f"Top 10 most frequent tokens: {list(zip(top_tokens.indices.tolist(), top_tokens.values.tolist()))}")

    # Filter and add silence
    speech_tokens_filtered = speech_tokens[speech_tokens < 6561]
    print(f"Filtered tokens: {len(speech_tokens_filtered)} (removed {len(speech_tokens) - len(speech_tokens_filtered)})")

    del model
    torch.cuda.empty_cache()


def test_vocoder_only():
    """Test S3Gen vocoder directly with simple token patterns."""
    print("\n" + "=" * 60)
    print("TEST 6: Vocoder-only test (bypass T3)")
    print("=" * 60)

    from chatterbox.tts_turbo import ChatterboxTurboTTS

    print("Loading model...")
    model = ChatterboxTurboTTS.from_pretrained(
        device="cuda",
        dtype="float32",
        compile_mode=None,
    )

    # Prepare conditionals (needed for S3Gen)
    model.prepare_conditionals(audio_prompt_path=None)

    # Test with simple repeating token patterns
    # This bypasses T3 entirely to test if vocoder works
    print("\nTesting vocoder with synthetic token patterns...")

    # Pattern 1: Silence token repeated
    from chatterbox.models.s3gen.const import S3GEN_SR
    silence_token = 6560  # Common silence token
    tokens = torch.tensor([silence_token] * 50).to(model.device)

    wav, _ = model.s3gen.inference(
        speech_tokens=tokens,
        ref_dict=model.conds.gen,
        n_cfm_timesteps=2,
    )
    wav = wav.squeeze(0).cpu()
    save_audio(wav.unsqueeze(0), model.sr, "test_vocoder_silence.wav")

    # Pattern 2: Random tokens in valid range
    random_tokens = torch.randint(0, 6500, (100,)).to(model.device)
    wav, _ = model.s3gen.inference(
        speech_tokens=random_tokens,
        ref_dict=model.conds.gen,
        n_cfm_timesteps=2,
    )
    wav = wav.squeeze(0).cpu()
    save_audio(wav.unsqueeze(0), model.sr, "test_vocoder_random.wav")

    # Pattern 3: Ascending tokens (should produce varying sounds)
    ascending_tokens = torch.arange(0, 200, 2).to(model.device)
    wav, _ = model.s3gen.inference(
        speech_tokens=ascending_tokens,
        ref_dict=model.conds.gen,
        n_cfm_timesteps=2,
    )
    wav = wav.squeeze(0).cpu()
    save_audio(wav.unsqueeze(0), model.sr, "test_vocoder_ascending.wav")

    print("\nVocoder tests saved. If these sound distorted too, the issue is in S3Gen.")
    print("If these sound clean but speech is distorted, the issue is in T3 token generation.")

    del model
    torch.cuda.empty_cache()


def main():
    print("=" * 60)
    print("MODEL DIAGNOSIS")
    print("=" * 60)
    print(f"Output directory: {output_dir}")

    # Run tests
    test_float32()
    test_float16()
    test_different_params()
    check_intermediate_outputs()
    test_vocoder_only()

    # CPU test is slow, skip by default
    # test_cpu()

    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)
    print(f"\nTest files saved to: {output_dir}")
    print("\nCompare the audio files:")
    print("  - test_float32.wav vs test_float16.wav (precision issue?)")
    print("  - test_params_*.wav (generation params issue?)")
    print("  - test_vocoder_*.wav (vocoder vs T3 issue?)")
    print("\nIf ALL files have the same distortion, the issue is likely:")
    print("  1. Model weights are corrupted")
    print("  2. S3Gen vocoder has issues")
    print("  3. Something in the base model architecture")


if __name__ == "__main__":
    main()
