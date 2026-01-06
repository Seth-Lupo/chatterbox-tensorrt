#!/usr/bin/env python3
"""
TensorRT-LLM Inference Test for T3 TTS

This script tests the full TTS pipeline using the TensorRT-LLM engine
with UNIFIED EMBEDDING architecture.

UNIFIED EMBEDDING LAYOUT:
    [0, speech_vocab_size)     = speech tokens (output, generated)
    [speech_vocab_size, total) = text tokens (input, offset)

This enables TRT-LLM's native generation loop:
    - Text input: original_text_id + speech_vocab_size (offset at input)
    - Speech output: 0 to speech_vocab_size-1 (feed directly back)

Usage:
    python test_trt_inference.py --engine_dir ./t3_engine --export_dir ./t3_export --output output_trt.wav
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import wavfile as scipy_wav

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class T3TRTInference:
    """
    TensorRT-LLM inference wrapper for T3 TTS with UNIFIED EMBEDDING.

    NEW Architecture (voice prefix as virtual tokens):
    - Unified vocab_embedding layout:
        [0, 251):        voice_prefix embeddings (virtual tokens)
        [251, 6814):     speech_emb
        [6814, 57090):   text_emb
    - lm_head outputs logits with speech at [251, 6814)
    - Voice prefix is GUARANTEED by prepending tokens [0, 1, ..., 250] to input

    This embeds voice conditioning directly into the input sequence.
    """

    def __init__(
        self,
        engine_dir: str,
        export_dir: str,
        device: str = "cuda",
    ):
        self.device = device
        self.engine_dir = Path(engine_dir)
        self.export_dir = Path(export_dir)

        # Load T3 metadata
        metadata_path = self.export_dir / "t3_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded T3 metadata: {self.metadata}")
        else:
            # Fallback to config.json
            with open(self.export_dir / "config.json") as f:
                config = json.load(f)
            self.metadata = config.get("t3_config", {})
            logger.info(f"Using t3_config from config.json: {self.metadata}")

        # Extract key parameters (NEW layout)
        self.voice_prefix_len = self.metadata.get("voice_prefix_len", 251)
        self.speech_vocab_size = self.metadata.get("speech_vocab_size", 6563)
        self.text_vocab_size = self.metadata.get("text_vocab_size", 50276)
        self.speech_offset = self.metadata.get("speech_offset", 251)
        self.text_offset = self.metadata.get("text_offset", 6814)
        self.stop_speech_token = self.metadata.get("stop_speech_token", 6562 + 251)
        self.start_speech_token = self.metadata.get("start_speech_token", 6561 + 251)

        logger.info(f"Unified embedding layout (voice prefix as virtual tokens):")
        logger.info(f"  [0, {self.voice_prefix_len}): voice_prefix (virtual tokens)")
        logger.info(f"  [{self.speech_offset}, {self.speech_offset + self.speech_vocab_size}): speech_emb")
        logger.info(f"  [{self.text_offset}, {self.text_offset + self.text_vocab_size}): text_emb")

        # Create voice prefix token IDs (virtual tokens 0, 1, 2, ..., 250)
        self.voice_prefix_tokens = torch.arange(
            self.voice_prefix_len, dtype=torch.int32, device=device
        )
        logger.info(f"Voice prefix tokens: [0, {self.voice_prefix_len}) = {self.voice_prefix_len} tokens")

        # Load TRT-LLM
        from tensorrt_llm.runtime import ModelRunner

        logger.info(f"Loading TRT-LLM engine from {engine_dir}")
        self.runner = ModelRunner.from_dir(engine_dir, rank=0)

        logger.info("T3 TRT Inference ready (UNIFIED EMBEDDING with voice prefix as virtual tokens)")

    def offset_text_tokens(self, text_token_ids: torch.Tensor) -> torch.Tensor:
        """
        Offset text tokens for unified embedding.

        Text tokens are stored at indices [text_offset, text_offset + text_vocab),
        so we add text_offset (6814) to the original text token IDs.
        """
        return text_token_ids + self.text_offset

    @torch.inference_mode()
    def generate(
        self,
        text_token_ids: torch.Tensor,
        max_new_tokens: int = 500,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        """
        Generate speech tokens from text using TRT-LLM with UNIFIED EMBEDDING.

        NEW Architecture:
        - Voice prefix is embedded as virtual tokens [0, 251)
        - Text tokens are offset to [6814, 57090)
        - Generated speech tokens are in [251, 6814)

        Flow:
        1. Prepend voice prefix tokens [0, 1, ..., 250]
        2. Offset and append text tokens
        3. TRT-LLM generates
        4. Output tokens in [251, 6814) are speech - subtract 251 to get actual token
        """
        logger.info(f"Generating speech from {text_token_ids.shape[1]} text tokens...")

        # Ensure input is on correct device and format
        if text_token_ids.dim() == 1:
            text_token_ids = text_token_ids.unsqueeze(0)

        # === OFFSET TEXT TOKENS ===
        offset_text = self.offset_text_tokens(text_token_ids[0])  # 1D tensor
        logger.info(f"Original text tokens (first 10): {text_token_ids[0, :10].tolist()}")
        logger.info(f"Offset text tokens (first 10): {offset_text[:10].tolist()}")

        # === PREPEND VOICE PREFIX VIRTUAL TOKENS ===
        # Voice prefix tokens [0, 1, 2, ..., 250] look up the baked voice embeddings
        full_input = torch.cat([self.voice_prefix_tokens, offset_text.int().cuda()])
        logger.info(f"Full input: {self.voice_prefix_len} voice tokens + {len(offset_text)} text tokens = {len(full_input)} total")

        # TRT-LLM expects batch_input_ids as a LIST of 1D tensors
        batch_input_ids = [full_input]

        # TRT-LLM generate - NO prompt_table needed, voice is in input_ids!
        output = self.runner.generate(
            batch_input_ids=batch_input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            end_id=self.stop_speech_token,  # 6562 + 251 = 6813
            pad_id=0,
        )

        # Extract output
        logger.info(f"Output type: {type(output)}")

        if isinstance(output, torch.Tensor):
            generated = output[0].cpu().tolist()
        elif hasattr(output, 'output_ids'):
            output_ids = output.output_ids
            if isinstance(output_ids, torch.Tensor):
                generated = output_ids[0].cpu().tolist()
            elif isinstance(output_ids, list):
                generated = output_ids[0]
                if isinstance(generated, torch.Tensor):
                    generated = generated.cpu().tolist()
                elif isinstance(generated, list) and len(generated) > 0 and isinstance(generated[0], list):
                    generated = generated[0]
        elif isinstance(output, dict) and 'output_ids' in output:
            generated = output['output_ids'][0]
            if hasattr(generated, 'cpu'):
                generated = generated.cpu().tolist()
        else:
            logger.warning(f"Unknown output format: {type(output)}")
            generated = []

        # Flatten if still nested
        if isinstance(generated, list) and len(generated) > 0 and isinstance(generated[0], list):
            generated = generated[0]

        logger.info(f"Generated {len(generated)} total tokens")
        logger.info(f"Raw tokens (first 30): {generated[:30]}")

        # Filter: keep only speech tokens in range [speech_offset, speech_offset + speech_vocab)
        # Then subtract speech_offset to get actual speech token IDs
        speech_tokens = []
        for t in generated:
            if self.speech_offset <= t < self.speech_offset + self.speech_vocab_size:
                actual_speech_token = t - self.speech_offset
                speech_tokens.append(actual_speech_token)

        logger.info(f"Filtered to {len(speech_tokens)} speech tokens (range [{self.speech_offset}, {self.speech_offset + self.speech_vocab_size}))")
        logger.info(f"Speech tokens (first 20): {speech_tokens[:20] if len(speech_tokens) > 20 else speech_tokens}")

        return torch.tensor(speech_tokens, dtype=torch.long, device=self.device)


class T3TRTInferenceSimple:
    """
    Simplified TRT inference that loads engine directly with TensorRT.
    Falls back to this if TRT-LLM ModelRunner doesn't work well with custom setup.
    """

    def __init__(
        self,
        engine_dir: str,
        export_dir: str,
        device: str = "cuda",
    ):
        import tensorrt as trt

        self.device = device
        self.engine_dir = Path(engine_dir)
        self.export_dir = Path(export_dir)

        # Load TensorRT engine
        logger.info(f"Loading TensorRT engine...")
        engine_path = self.engine_dir / "rank0.engine"

        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(trt_logger).deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Load custom components
        logger.info("Loading custom components...")

        self.voice_prefix = torch.from_numpy(
            np.load(self.export_dir / "prompt_table.npy")
        ).to(device).half()

        self.speech_embedding = torch.from_numpy(
            np.load(self.export_dir / "speech_embedding.npy")
        ).to(device).half()

        speech_head = np.load(self.export_dir / "speech_head.npz")
        self.speech_head_weight = torch.from_numpy(speech_head["weight"]).to(device).half()
        self.speech_head_bias = torch.from_numpy(speech_head["bias"]).to(device).half()

        import json
        with open(self.export_dir / "config.json") as f:
            self.config = json.load(f)

        self.speech_vocab_size = self.config["t3_config"]["speech_vocab_size"]
        self.start_speech_token = 6561
        self.stop_speech_token = 6562

        logger.info("T3 TRT Simple Inference ready")

    def apply_speech_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply speech output head."""
        return F.linear(hidden_states, self.speech_head_weight, self.speech_head_bias)

    @torch.inference_mode()
    def generate_simple(
        self,
        text: str,
        tokenizer,
        max_new_tokens: int = 500,
        temperature: float = 0.8,
    ) -> torch.Tensor:
        """
        Simple generation - for testing the pipeline.
        Note: This is a placeholder - full TRT inference requires proper I/O binding.
        """
        logger.warning("Simple inference not fully implemented - use T3TRTInference")
        return torch.zeros(100, dtype=torch.long, device=self.device)


def test_full_pipeline(
    engine_dir: str,
    export_dir: str,
    output_path: str,
    text: str = "This is a test of the TensorRT accelerated text to speech system.",
):
    """Test the full TTS pipeline with TRT engine."""

    logger.info("=" * 60)
    logger.info("T3 TensorRT-LLM Full Pipeline Test")
    logger.info("=" * 60)

    # Load tokenizer
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    import os

    logger.info("Loading tokenizer...")
    local_path = snapshot_download(
        repo_id="ResembleAI/chatterbox-turbo",
        token=os.getenv("HF_TOKEN") or True,
        allow_patterns=["*.json", "*.txt", "*.model"]
    )
    tokenizer = AutoTokenizer.from_pretrained(local_path)

    # Tokenize text
    text_tokens = tokenizer(text, return_tensors="pt").input_ids.cuda()
    logger.info(f"Text: '{text}'")
    logger.info(f"Text tokens: {text_tokens.shape}")

    # Try TRT-LLM inference
    try:
        logger.info("Attempting TRT-LLM ModelRunner inference...")
        model = T3TRTInference(engine_dir, export_dir)

        start_time = time.time()
        speech_tokens = model.generate(
            text_tokens,
            max_new_tokens=500,
            temperature=0.8,
        )
        gen_time = time.time() - start_time

        logger.info(f"Generated {len(speech_tokens)} speech tokens in {gen_time:.2f}s")

    except Exception as e:
        logger.error(f"TRT-LLM inference failed: {e}")
        logger.info("Falling back to PyTorch model for vocoder test...")

        # Fall back to PyTorch model to test vocoder
        from src.chatterbox.tts_turbo_trt import ChatterboxTurboTRT

        model = ChatterboxTurboTRT.from_pretrained(
            device="cuda",
            voice_audio_path="voice_ref.wav"
        )

        start_time = time.time()
        audio = model.generate(text)
        gen_time = time.time() - start_time

        logger.info(f"Generated audio in {gen_time:.2f}s")
        logger.info(f"Audio shape: {audio.shape}")

        # Save audio
        audio_np = audio.squeeze().cpu().numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)
        scipy_wav.write(output_path, 24000, audio_int16)
        logger.info(f"Saved audio to {output_path}")

        return

    # If TRT worked, run vocoder
    logger.info("Running S3Gen vocoder...")

    from safetensors.torch import load_file
    from src.chatterbox.models.s3gen import S3Gen, S3GEN_SR
    from src.chatterbox.models.s3gen.const import S3GEN_SIL

    # Load S3Gen
    s3gen = S3Gen(meanflow=True)
    s3gen.load_state_dict(load_file(Path(local_path) / "s3gen_meanflow.safetensors"))
    s3gen.to("cuda").eval()

    # Load reference for S3Gen (reuse voice_ref.wav)
    orig_sr, wav_np = scipy_wav.read("voice_ref.wav")
    if wav_np.dtype == np.int16:
        wav_np = wav_np.astype(np.float32) / 32768.0
    elif wav_np.dtype == np.int32:
        wav_np = wav_np.astype(np.float32) / 2147483648.0
    elif wav_np.dtype != np.float32:
        wav_np = wav_np.astype(np.float32)

    # Convert stereo to mono
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=1)

    if orig_sr != S3GEN_SR:
        from scipy import signal as scipy_signal
        num_samples = int(len(wav_np) * S3GEN_SR / orig_sr)
        wav_np = scipy_signal.resample(wav_np, num_samples)

    wav = torch.from_numpy(wav_np.astype(np.float32)).cuda()

    s3gen_ref = s3gen.embed_ref(wav[:10 * S3GEN_SR], S3GEN_SR, device="cuda")

    # Filter and prepare speech tokens
    speech_tokens = speech_tokens[speech_tokens < 6561]
    silence = torch.tensor([S3GEN_SIL] * 3, dtype=torch.long, device="cuda")
    speech_tokens = torch.cat([speech_tokens, silence])

    # Generate audio
    logger.info(f"Vocoding {len(speech_tokens)} speech tokens...")
    audio, _ = s3gen.inference(
        speech_tokens=speech_tokens,
        ref_dict=s3gen_ref,
        n_cfm_timesteps=2,
    )

    # Save
    audio_np = audio.squeeze().cpu().numpy()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    scipy_wav.write(output_path, S3GEN_SR, audio_int16)

    logger.info(f"Saved audio to {output_path}")
    logger.info("=" * 60)
    logger.info("Test complete!")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test T3 TRT-LLM Inference")
    parser.add_argument("--engine_dir", type=str, default="./t3_engine")
    parser.add_argument("--export_dir", type=str, default="./t3_export")
    parser.add_argument("--output", "-o", type=str, default="output_trt.wav")
    parser.add_argument("--text", "-t", type=str,
                        default="Hello my friend! This is a test of the TensorRT accelerated text to speech system.")

    args = parser.parse_args()

    test_full_pipeline(
        args.engine_dir,
        args.export_dir,
        args.output,
        args.text,
    )


if __name__ == "__main__":
    main()
