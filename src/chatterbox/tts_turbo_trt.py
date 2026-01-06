"""
Chatterbox Turbo TTS - TensorRT-LLM Compatible Version

This is a modified version of ChatterboxTurboTTS that uses the T3ForTRT model
with BAKED voice conditioning at initialization time. This enables:
- TensorRT-LLM export compatibility (input_ids only, no inputs_embeds)
- Single voice per model instance (voice baked at init)
- Faster inference with static conditioning

Usage:
    # Create TTS with baked voice
    model = ChatterboxTurboTRT.from_pretrained(
        device="cuda",
        voice_audio_path="/path/to/reference.wav"
    )

    # Generate audio (voice is already baked)
    audio = model.generate("Hello, world!")
"""

import os
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import wavfile as scipy_wav
from scipy import signal as scipy_signal
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, GPT2Model, GPT2Config
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from .models.s3tokenizer import S3_SR
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond, T3CondEnc
from .models.t3.modules.t3_config import T3Config
from .models.s3gen.const import S3GEN_SIL

logger = logging.getLogger(__name__)


# =============================================================================
# Inlined utilities (avoid torchaudio dependency from tts_turbo)
# =============================================================================

def setup_cuda_optimizations():
    """Enable CUDA optimizations for faster inference."""
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)


def normalize_text(text: str) -> str:
    """Normalize text for TTS: fix punctuation and capitalization."""
    if len(text) == 0:
        return "You need to add some text for me to talk."
    if text[0].islower():
        text = text[0].upper() + text[1:]
    text = " ".join(text.split())
    replacements = [
        ("…", ", "), (":", ","), ("—", "-"), ("–", "-"), (" ,", ","),
        (""", "\""), (""", "\""), ("'", "'"), ("'", "'"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    text = text.rstrip(" ")
    if not any(text.endswith(p) for p in {".", "!", "?", "-", ","}):
        text += "."
    return text


@dataclass
class StreamingMetrics:
    """Metrics for streaming TTS generation."""
    latency_to_first_chunk: Optional[float] = None
    rtf: Optional[float] = None
    total_generation_time: Optional[float] = None
    total_audio_duration: Optional[float] = None
    chunk_count: int = 0
    cumulative_gen_time: float = 0.0
    cumulative_audio_duration: float = 0.0
    buffer_ahead: float = 0.0

REPO_ID = "ResembleAI/chatterbox-turbo"


# =============================================================================
# T3ForTRT - TensorRT-LLM Compatible T3 for Turbo
# =============================================================================

@dataclass
class T3TurboTRTConfig:
    """Configuration for T3 Turbo TRT model (GPT2-based)."""

    # Text tokens (GPT2 vocabulary)
    text_vocab_size: int = 50276

    # Speech tokens
    speech_vocab_size: int = 6563
    start_speech_token: int = 6561
    stop_speech_token: int = 6562

    # Model dimensions (GPT2-medium)
    hidden_size: int = 1024
    num_layers: int = 24
    num_heads: int = 16

    # Conditioning
    speaker_embed_size: int = 256
    speech_cond_prompt_len: int = 375

    # Voice prefix length (1 speaker + up to 375 prompt tokens)
    voice_prefix_len: int = 376

    @property
    def total_vocab_size(self) -> int:
        return self.text_vocab_size + self.speech_vocab_size

    @property
    def unified_start_speech(self) -> int:
        return self.text_vocab_size + self.start_speech_token

    @property
    def unified_stop_speech(self) -> int:
        return self.text_vocab_size + self.stop_speech_token


class T3TurboForTRT(torch.nn.Module):
    """
    TensorRT-LLM compatible T3 Turbo model.

    Key differences from original T3:
    - Voice conditioning is BAKED at initialization (not runtime)
    - Uses unified embedding table (text + speech)
    - Accepts only input_ids (no inputs_embeds at runtime)
    - No perceiver resampler or emotion adversarial

    Architecture:
        [BAKED_VOICE_PREFIX] + [TEXT_TOKENS] + [SPEECH_TOKENS (autoregressive)]
    """

    def __init__(self, config: T3TurboTRTConfig):
        super().__init__()
        self.config = config

        # GPT2-medium backbone
        # Use small dummy vocab - we'll use custom embeddings instead
        gpt2_config = GPT2Config(
            vocab_size=8,  # Dummy - we use custom embeddings
            n_positions=8196,
            n_embd=config.hidden_size,
            n_layer=config.num_layers,
            n_head=config.num_heads,
            activation_function="gelu_new",
            attn_pdrop=0.0,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            use_cache=True,
        )
        self.transformer = GPT2Model(gpt2_config)

        # Delete GPT2's built-in embeddings - we use custom ones
        del self.transformer.wte

        # Separate embedding tables (will be unified for export)
        self.text_emb = torch.nn.Embedding(config.text_vocab_size, config.hidden_size)
        self.speech_emb = torch.nn.Embedding(config.speech_vocab_size, config.hidden_size)

        # Output head (speech logits only)
        self.speech_head = torch.nn.Linear(config.hidden_size, config.speech_vocab_size, bias=True)

        # Conditioning encoder (for baking voice prefix)
        self.spkr_enc = torch.nn.Linear(config.speaker_embed_size, config.hidden_size)

        # Voice prefix buffer (set by bake_voice_conditioning)
        self.register_buffer("_voice_prefix", None)
        self._voice_prefix_len = 0

    @property
    def device(self):
        return self.speech_head.weight.device

    @property
    def dtype(self):
        return self.speech_head.weight.dtype

    @property
    def voice_prefix(self) -> torch.Tensor:
        if self._voice_prefix is None:
            raise RuntimeError("Voice not baked. Call bake_voice_conditioning() first.")
        return self._voice_prefix

    def bake_voice_conditioning(
        self,
        speaker_emb: torch.Tensor,
        cond_prompt_speech_tokens: torch.Tensor,
    ):
        """
        Bake voice conditioning into the model as a fixed prefix.

        This extracts the conditioning embeddings and stores them as a buffer.
        The voice cannot be changed after baking without re-calling this method.

        Args:
            speaker_emb: Speaker embedding from voice encoder, shape (256,) or (1, 256)
            cond_prompt_speech_tokens: Speech tokens from reference audio, shape (1, T)
        """
        with torch.no_grad():
            # Ensure proper shapes
            speaker_emb = speaker_emb.to(device=self.device, dtype=self.dtype)
            if speaker_emb.dim() == 1:
                speaker_emb = speaker_emb.unsqueeze(0)

            cond_prompt_speech_tokens = cond_prompt_speech_tokens.to(device=self.device)
            if cond_prompt_speech_tokens.dim() == 1:
                cond_prompt_speech_tokens = cond_prompt_speech_tokens.unsqueeze(0)

            # Project speaker embedding: (1, 256) -> (1, 1, 1024)
            speaker_prefix = self.spkr_enc(speaker_emb).unsqueeze(1)  # (1, 1, hidden)

            # Embed speech prompt tokens: (1, T) -> (1, T, 1024)
            speech_prefix = self.speech_emb(cond_prompt_speech_tokens)  # (1, T, hidden)

            # Concatenate: [speaker, speech_prompt]
            voice_prefix = torch.cat([speaker_prefix, speech_prefix], dim=1)  # (1, 1+T, hidden)

            # Store as buffer
            self._voice_prefix = voice_prefix
            self._voice_prefix_len = voice_prefix.size(1)

            logger.info(f"Baked voice conditioning: shape={voice_prefix.shape}")

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed input_ids from unified token space.

        Token ranges:
            [0, text_vocab_size) -> text embeddings
            [text_vocab_size, total_vocab_size) -> speech embeddings
        """
        B, seq_len = input_ids.shape
        is_text = input_ids < self.config.text_vocab_size
        is_speech = ~is_text

        embeddings = torch.zeros(
            B, seq_len, self.config.hidden_size,
            device=input_ids.device, dtype=self.dtype
        )

        # Embed text tokens
        if is_text.any():
            text_ids = input_ids.clone()
            text_ids[is_speech] = 0
            text_embeds = self.text_emb(text_ids)
            embeddings = torch.where(
                is_text.unsqueeze(-1).expand_as(embeddings),
                text_embeds,
                embeddings
            )

        # Embed speech tokens (offset by text_vocab_size)
        if is_speech.any():
            speech_ids = (input_ids - self.config.text_vocab_size).clamp(min=0)
            speech_ids[is_text] = 0
            speech_embeds = self.speech_emb(speech_ids)
            embeddings = torch.where(
                is_speech.unsqueeze(-1).expand_as(embeddings),
                speech_embeds,
                embeddings
            )

        return embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = True,
        prepend_voice_prefix: bool = True,
    ):
        """
        Forward pass with baked voice prefix.

        For initial context: input_ids are text tokens, voice prefix prepended
        For generation: input_ids are speech tokens, use KV cache
        """
        # Get embeddings
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Provide input_ids or inputs_embeds")
            inputs_embeds = self.embed_input_ids(input_ids)

        # Prepend voice prefix for initial pass
        if prepend_voice_prefix and past_key_values is None:
            voice_prefix = self.voice_prefix.to(dtype=inputs_embeds.dtype)
            if voice_prefix.size(0) != inputs_embeds.size(0):
                voice_prefix = voice_prefix.expand(inputs_embeds.size(0), -1, -1)
            inputs_embeds = torch.cat([voice_prefix, inputs_embeds], dim=1)

        # Transformer forward
        outputs = self.transformer(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )

        # Project to speech logits
        logits = self.speech_head(outputs.last_hidden_state)

        return {
            "logits": logits,
            "past_key_values": outputs.past_key_values if use_cache else None,
            "hidden_states": outputs.last_hidden_state,
        }


# =============================================================================
# Weight Loading Utilities
# =============================================================================

def load_t3_turbo_weights_to_trt(
    t3_trt_model: T3TurboForTRT,
    t3_state_dict: dict,
) -> T3TurboForTRT:
    """
    Load original T3 Turbo weights into T3TurboForTRT model.

    Key mappings:
        tfmr.* -> transformer.* (excluding wte)
        text_emb.* -> text_emb.*
        speech_emb.* -> speech_emb.*
        speech_head.* -> speech_head.*
        cond_enc.spkr_enc.* -> spkr_enc.*
    """
    new_state_dict = {}

    for key, value in t3_state_dict.items():
        if key.startswith("tfmr."):
            # Map transformer weights, but skip wte (we use custom embeddings)
            if key.startswith("tfmr.wte."):
                continue  # Skip - we use custom text_emb instead
            new_key = "transformer." + key[5:]
            new_state_dict[new_key] = value
        elif key.startswith("text_emb."):
            new_state_dict[key] = value
        elif key.startswith("speech_emb."):
            new_state_dict[key] = value
        elif key.startswith("speech_head."):
            new_state_dict[key] = value
        elif key.startswith("cond_enc.spkr_enc."):
            # Map speaker encoder
            new_key = "spkr_enc." + key[18:]
            new_state_dict[new_key] = value
        # Skip other cond_enc weights (perceiver, emotion_adv) - not used in Turbo

    # Load with strict=False to handle any remaining mismatches
    missing, unexpected = t3_trt_model.load_state_dict(new_state_dict, strict=False)

    logger.info(f"Loaded T3 weights: {len(new_state_dict)} keys")
    if missing:
        # Filter out expected missing keys
        unexpected_missing = [k for k in missing if not k.startswith("transformer.wte")]
        if unexpected_missing:
            logger.warning(f"Missing keys: {unexpected_missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")

    return t3_trt_model


# =============================================================================
# Main TTS Class - TRT Compatible
# =============================================================================

class ChatterboxTurboTRT:
    """
    TensorRT-LLM compatible Chatterbox Turbo TTS.

    Key differences from ChatterboxTurboTTS:
    - Voice is BAKED at initialization (one voice per instance)
    - Uses T3TurboForTRT model (input_ids only, no inputs_embeds)
    - Cannot change voice without creating new instance

    This enables TensorRT-LLM export where:
    - Voice conditioning is in the prompt table (baked)
    - Text tokens are dynamic input_ids
    - Speech tokens are generated autoregressively
    """

    ENC_COND_LEN = 15 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3TurboForTRT,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer,
        device: str,
        s3gen_ref_dict: dict,
    ):
        self.sr = S3GEN_SR
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.s3gen_ref_dict = s3gen_ref_dict  # Baked S3Gen conditioning
        self._logits_processors_cache = {}
        self._resamplers = {}

    @classmethod
    def from_pretrained(
        cls,
        device: str,
        voice_audio_path: str,
        exaggeration: float = 0.5,
    ) -> 'ChatterboxTurboTRT':
        """
        Load model with BAKED voice conditioning.

        Args:
            device: Device to load model on
            voice_audio_path: Path to reference audio for voice cloning
            exaggeration: Emotion exaggeration factor (for S3Gen)

        Returns:
            ChatterboxTurboTRT instance with baked voice
        """
        if device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS not available, falling back to CPU")
            device = "cpu"

        local_path = snapshot_download(
            repo_id=REPO_ID,
            token=os.getenv("HF_TOKEN") or True,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"]
        )
        return cls.from_local(local_path, device, voice_audio_path, exaggeration)

    @classmethod
    def from_local(
        cls,
        ckpt_dir: str,
        device: str,
        voice_audio_path: str,
        exaggeration: float = 0.5,
    ) -> 'ChatterboxTurboTRT':
        """Load model from local checkpoint with baked voice."""
        ckpt_dir = Path(ckpt_dir)

        if device == "cuda":
            setup_cuda_optimizations()

        # Load voice encoder
        ve = VoiceEncoder()
        ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
        ve.to(device).eval()

        # Create T3TurboForTRT model
        config = T3TurboTRTConfig()
        t3 = T3TurboForTRT(config)

        # Load T3 weights
        t3_state = load_file(ckpt_dir / "t3_turbo_v1.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        load_t3_turbo_weights_to_trt(t3, t3_state)
        t3.to(device=device).eval()

        # Load S3Gen
        s3gen = S3Gen(meanflow=True)
        s3gen.load_state_dict(load_file(ckpt_dir / "s3gen_meanflow.safetensors"), strict=True)
        s3gen.to(device=device).eval()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # === BAKE VOICE CONDITIONING ===
        logger.info(f"Baking voice from: {voice_audio_path}")

        # Load and preprocess audio
        orig_sr, wav_np = scipy_wav.read(voice_audio_path)
        if wav_np.dtype == np.int16:
            wav_np = wav_np.astype(np.float32) / 32768.0
        elif wav_np.dtype == np.int32:
            wav_np = wav_np.astype(np.float32) / 2147483648.0
        elif wav_np.dtype != np.float32:
            wav_np = wav_np.astype(np.float32)
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=1)

        # Resample to 24kHz using scipy (avoids torchaudio/ffmpeg dependency)
        if orig_sr != S3GEN_SR:
            num_samples = int(len(wav_np) * S3GEN_SR / orig_sr)
            wav_np = scipy_signal.resample(wav_np, num_samples)

        assert len(wav_np) / S3GEN_SR > 5.0, "Audio prompt must be longer than 5 seconds!"

        wav = torch.from_numpy(wav_np.astype(np.float32)).to(device)

        # Resample to 16kHz for encoder using scipy
        num_samples_16k = int(len(wav_np) * S3_SR / S3GEN_SR)
        ref_16k_np = scipy_signal.resample(wav_np, num_samples_16k)
        ref_16k = torch.from_numpy(ref_16k_np.astype(np.float32)).to(device)

        # Get S3Gen reference embeddings (for vocoder)
        s3gen_ref = s3gen.embed_ref(wav[:cls.DEC_COND_LEN], S3GEN_SR, device=device)

        # Get T3 conditioning tokens
        t3_tokens, _ = s3gen.tokenizer.forward(
            [ref_16k[:cls.ENC_COND_LEN]],
            max_len=config.speech_cond_prompt_len
        )
        t3_tokens = torch.atleast_2d(t3_tokens).to(device)

        # Get speaker embedding
        ve_embed = torch.from_numpy(
            ve.embeds_from_wavs([ref_16k.cpu().numpy()], sample_rate=S3_SR)
        ).mean(axis=0, keepdim=True).to(device)

        # BAKE voice into T3
        t3.bake_voice_conditioning(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_tokens,
        )

        logger.info(f"Voice baked! Prefix length: {t3._voice_prefix_len} tokens")

        return cls(t3, s3gen, ve, tokenizer, device, s3gen_ref)

    def generate(
        self,
        text: str,
        temperature: float = 0.8,
        top_k: int = 1000,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        max_tokens: int = 1000,
    ) -> torch.Tensor:
        """
        Generate speech from text using baked voice.

        Args:
            text: Input text to synthesize
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty
            max_tokens: Maximum speech tokens to generate

        Returns:
            Audio tensor of shape (1, samples)
        """
        text = normalize_text(text)
        text_tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        text_tokens = text_tokens.input_ids.to(self.device)

        # Generate speech tokens
        speech_tokens = self._generate_speech_tokens(
            text_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
        )

        # Filter and add silence
        speech_tokens = speech_tokens[speech_tokens < 6561].to(self.device)
        silence = torch.tensor([S3GEN_SIL] * 3, dtype=torch.long, device=self.device)
        speech_tokens = torch.cat([speech_tokens, silence])

        # Vocoder
        wav, _ = self.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=self.s3gen_ref_dict,
            n_cfm_timesteps=2,
        )

        return wav.squeeze(0).detach().cpu().unsqueeze(0)

    @torch.inference_mode()
    def _generate_speech_tokens(
        self,
        text_tokens: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        max_tokens: int,
    ) -> torch.Tensor:
        """Generate speech tokens autoregressively using baked voice."""

        processors = self._get_logits_processors(temperature, top_k, top_p, repetition_penalty)

        # Get embeddings for text (voice prefix prepended automatically)
        text_embeds = self.t3.text_emb(text_tokens)  # (1, T, hidden)

        # Initial forward pass with voice prefix + text
        output = self.t3.forward(
            inputs_embeds=text_embeds,
            use_cache=True,
            prepend_voice_prefix=True,
        )
        kv_cache = output["past_key_values"]

        # Get first speech token logits (last position)
        logits = output["logits"][:, -1, :]  # (1, speech_vocab)

        # Initialize token storage
        generated = []
        start_token = torch.tensor([[self.t3.config.start_speech_token]], device=self.device)

        # Apply processors and sample first token
        logits = processors(start_token, logits)
        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        generated.append(token)

        stop_token = self.t3.config.stop_speech_token

        # Generation loop
        for _ in range(max_tokens - 1):
            if token.item() == stop_token:
                break

            # Embed current token
            token_embed = self.t3.speech_emb(token)  # (1, 1, hidden)

            # Forward with KV cache
            output = self.t3.forward(
                inputs_embeds=token_embed,
                past_key_values=kv_cache,
                use_cache=True,
                prepend_voice_prefix=False,
            )
            kv_cache = output["past_key_values"]
            logits = output["logits"][:, -1, :]

            # Build history for processors
            history = torch.cat(generated, dim=1)
            logits = processors(history, logits)

            if torch.all(logits == float("-inf")):
                break

            probs = F.softmax(logits, dim=-1)
            token = torch.multinomial(probs, num_samples=1)
            generated.append(token)

        # Concatenate all tokens
        all_tokens = torch.cat(generated, dim=1).squeeze(0)
        return all_tokens

    def _get_logits_processors(
        self, temperature: float, top_k: int, top_p: float, repetition_penalty: float
    ) -> LogitsProcessorList:
        """Get or create cached logits processors."""
        key = (temperature, top_k, top_p, repetition_penalty)
        if key not in self._logits_processors_cache:
            processors = LogitsProcessorList()
            if temperature > 0 and temperature != 1.0:
                processors.append(TemperatureLogitsWarper(temperature))
            if top_k > 0:
                processors.append(TopKLogitsWarper(top_k))
            if top_p < 1.0:
                processors.append(TopPLogitsWarper(top_p))
            if repetition_penalty != 1.0:
                processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
            self._logits_processors_cache[key] = processors
        return self._logits_processors_cache[key]

    def generate_stream(
        self,
        text: str,
        temperature: float = 0.8,
        top_k: int = 1000,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        crossfade_ms: float = 20.0,
        chunk_size: int = 32,
        max_tokens: int = 1000,
    ) -> Generator[Tuple[torch.Tensor, StreamingMetrics], None, None]:
        """
        Streaming TTS generation with baked voice.

        Yields audio chunks as they are generated.
        """
        start_time = time.time()
        metrics = StreamingMetrics()

        text = normalize_text(text)
        text_tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        text_tokens = text_tokens.input_ids.to(self.device)

        all_tokens = torch.tensor([], dtype=torch.long, device=self.device)
        chunk_buffer = []
        prev_tail = None

        with torch.inference_mode():
            for token in self._stream_tokens(text_tokens, temperature, top_k, top_p, repetition_penalty, max_tokens):
                chunk_buffer.append(token)

                if len(chunk_buffer) >= chunk_size:
                    new_tokens = torch.cat(chunk_buffer, dim=-1).squeeze(0)

                    audio, duration, success, prev_tail = self._process_chunk(
                        new_tokens, all_tokens, start_time, metrics, prev_tail, crossfade_ms
                    )

                    if success:
                        yield audio, metrics

                    all_tokens = torch.cat([all_tokens, new_tokens]) if len(all_tokens) > 0 else new_tokens
                    chunk_buffer = []

            # Process remaining
            if chunk_buffer:
                new_tokens = torch.cat(chunk_buffer, dim=-1).squeeze(0)
                audio, duration, success, prev_tail = self._process_chunk(
                    new_tokens, all_tokens, start_time, metrics, prev_tail, crossfade_ms
                )
                if success:
                    yield audio, metrics

            # Flush tail
            if prev_tail is not None and len(prev_tail) > 0:
                final_audio = torch.from_numpy(prev_tail.copy()).unsqueeze(0)
                yield final_audio, metrics

    def _stream_tokens(
        self,
        text_tokens: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        max_tokens: int,
    ) -> Generator[torch.Tensor, None, None]:
        """Stream speech tokens one at a time."""
        processors = self._get_logits_processors(temperature, top_k, top_p, repetition_penalty)

        # Initial forward with voice prefix + text
        text_embeds = self.t3.text_emb(text_tokens)
        output = self.t3.forward(inputs_embeds=text_embeds, use_cache=True, prepend_voice_prefix=True)
        kv_cache = output["past_key_values"]
        logits = output["logits"][:, -1, :]

        start_token = torch.tensor([[self.t3.config.start_speech_token]], device=self.device)
        logits = processors(start_token, logits)
        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)

        generated = [token]
        yield token

        stop_token = self.t3.config.stop_speech_token

        for _ in range(max_tokens - 1):
            if token.item() == stop_token:
                break

            token_embed = self.t3.speech_emb(token)
            output = self.t3.forward(inputs_embeds=token_embed, past_key_values=kv_cache, use_cache=True, prepend_voice_prefix=False)
            kv_cache = output["past_key_values"]
            logits = output["logits"][:, -1, :]

            history = torch.cat(generated, dim=1)
            logits = processors(history, logits)

            if torch.all(logits == float("-inf")):
                break

            probs = F.softmax(logits, dim=-1)
            token = torch.multinomial(probs, num_samples=1)
            generated.append(token)
            yield token

    def _process_chunk(
        self,
        new_tokens: torch.Tensor,
        all_tokens: torch.Tensor,
        start_time: float,
        metrics: StreamingMetrics,
        prev_tail: Optional[np.ndarray],
        crossfade_ms: float,
    ):
        """Process a chunk of tokens into audio."""
        chunk_start = time.time()

        # Context for S3Gen
        context_window = 60
        if len(all_tokens) > 0:
            context = all_tokens[-context_window:] if len(all_tokens) > context_window else all_tokens
            tokens = torch.cat([context, new_tokens])
            context_len = len(context)
        else:
            tokens = new_tokens
            context_len = 0

        # Filter invalid tokens
        tokens = tokens[tokens < 6561]
        if len(tokens) == 0:
            return None, 0.0, False, prev_tail

        # Generate audio
        wav, _ = self.s3gen.inference(
            speech_tokens=tokens.to(self.device),
            ref_dict=self.s3gen_ref_dict,
            n_cfm_timesteps=2,
        )
        wav = wav.squeeze(0).detach().cpu().numpy()

        # Crop context
        if context_len > 0:
            samples_per_token = len(wav) / len(tokens)
            skip = int(context_len * samples_per_token)
            audio = wav[skip:]
        else:
            audio = wav

        if len(audio) == 0:
            return None, 0.0, False, prev_tail

        # Crossfade
        crossfade_samples = int(crossfade_ms * self.sr / 1000)
        audio = audio - np.mean(audio)

        if prev_tail is not None and crossfade_samples > 0:
            blend_len = min(crossfade_samples, len(prev_tail), len(audio))
            if blend_len > 0:
                t = np.linspace(0, np.pi / 2, blend_len, dtype=audio.dtype)
                blended = prev_tail[:blend_len] * np.cos(t) + audio[:blend_len] * np.sin(t)
                audio = np.concatenate([blended, audio[blend_len:]])

        # Hold back tail
        if len(audio) > crossfade_samples:
            new_tail = audio[-crossfade_samples:].copy()
            audio = audio[:-crossfade_samples]
        else:
            new_tail = audio.copy()
            audio = np.array([], dtype=audio.dtype)

        audio_duration = len(audio) / self.sr
        audio_tensor = torch.from_numpy(audio.copy()).unsqueeze(0)

        # Metrics
        if metrics.chunk_count == 0:
            metrics.latency_to_first_chunk = time.time() - start_time
        metrics.chunk_count += 1

        return audio_tensor, audio_duration, True, new_tail

    def save_voice_prefix(self, path: str):
        """Save the baked voice prefix for TensorRT-LLM export."""
        if self.t3._voice_prefix is None:
            raise RuntimeError("No voice baked")

        torch.save({
            "voice_prefix": self.t3._voice_prefix.cpu(),
            "metadata": {
                "len_cond": self.t3._voice_prefix_len,
                "hidden_size": self.t3.config.hidden_size,
                "dtype": str(self.t3._voice_prefix.dtype),
            }
        }, path)
        logger.info(f"Saved voice prefix to {path}")

    def export_for_trtllm(self, output_dir: str):
        """
        Export model for TensorRT-LLM.

        Creates:
            - config.json: TRT-LLM compatible config
            - rank0.safetensors: Model weights in TRT-LLM format
            - prompt_table.npy: Baked voice prefix for ptuning
        """
        import json
        from safetensors.torch import save_file

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # TRT-LLM config format
        # vocab_size = text_vocab_size for input embedding
        # We handle speech output head separately (saved as speech_head.npy)
        config = {
            "architecture": "GPTForCausalLM",
            "dtype": "float16",
            "num_hidden_layers": self.t3.config.num_layers,
            "num_attention_heads": self.t3.config.num_heads,
            "hidden_size": self.t3.config.hidden_size,
            "intermediate_size": self.t3.config.hidden_size * 4,
            "num_key_value_heads": self.t3.config.num_heads,
            "vocab_size": self.t3.config.text_vocab_size,  # Input vocab (text)
            "position_embedding_type": "learned_absolute",
            "max_position_embeddings": 8196,
            "hidden_act": "gelu",
            "norm_epsilon": 1e-5,
            "share_embedding_table": True,  # Tie lm_head to vocab_embedding
            "mapping": {
                "world_size": 1,
                "tp_size": 1,
                "pp_size": 1
            },
            "quantization": {
                "quant_algo": None
            },
            # Custom fields for T3
            "t3_config": {
                "text_vocab_size": self.t3.config.text_vocab_size,
                "speech_vocab_size": self.t3.config.speech_vocab_size,
                "voice_prefix_len": self.t3._voice_prefix_len,
            }
        }
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved TRT-LLM config")

        # Save voice prefix as prompt table (for ptuning)
        voice_prefix = self.t3._voice_prefix.squeeze(0).cpu().half().numpy()
        np.save(output_dir / "prompt_table.npy", voice_prefix)
        logger.info(f"Saved prompt table: shape={voice_prefix.shape}")

        # Save speech embedding separately (for custom runtime handling)
        speech_emb = self.t3.speech_emb.weight.detach().cpu().half().numpy()
        np.save(output_dir / "speech_embedding.npy", speech_emb)
        logger.info(f"Saved speech embedding: shape={speech_emb.shape}")

        # Save speech head separately (TRT-LLM GPT uses tied lm_head, we need custom output)
        speech_head_w = self.t3.speech_head.weight.detach().cpu().half().numpy()
        speech_head_b = self.t3.speech_head.bias.detach().cpu().half().numpy()
        np.savez(output_dir / "speech_head.npz", weight=speech_head_w, bias=speech_head_b)
        logger.info(f"Saved speech head: weight={speech_head_w.shape}, bias={speech_head_b.shape}")

        # Convert weights to TRT-LLM naming convention
        trtllm_weights = self._convert_weights_to_trtllm()
        save_file(trtllm_weights, output_dir / "rank0.safetensors")
        logger.info(f"Saved TRT-LLM weights: {len(trtllm_weights)} tensors")

        logger.info(f"Exported TRT-LLM checkpoint to {output_dir}")
        return output_dir

    def _convert_weights_to_trtllm(self) -> dict:
        """Convert model weights to TRT-LLM naming convention for GPTForCausalLM."""
        trtllm_weights = {}

        # Get state dict
        state_dict = self.t3.state_dict()

        for name, param in state_dict.items():
            param = param.cpu().half().contiguous()

            # Skip voice prefix buffer (saved separately)
            if name == "_voice_prefix":
                continue

            # Map transformer weights
            # GPT2 in HF: transformer.h.{layer}.{component}
            # TRT-LLM GPT: transformer.layers.{layer}.{component}
            if name.startswith("transformer.h."):
                parts = name.split(".")
                layer_idx = parts[2]
                component = ".".join(parts[3:])

                # TRT-LLM GPT naming convention
                if component == "ln_1.weight":
                    new_name = f"transformer.layers.{layer_idx}.input_layernorm.weight"
                elif component == "ln_1.bias":
                    new_name = f"transformer.layers.{layer_idx}.input_layernorm.bias"
                elif component == "ln_2.weight":
                    new_name = f"transformer.layers.{layer_idx}.post_layernorm.weight"
                elif component == "ln_2.bias":
                    new_name = f"transformer.layers.{layer_idx}.post_layernorm.bias"
                elif component == "attn.c_attn.weight":
                    # GPT2 uses combined QKV, TRT-LLM expects it transposed
                    new_name = f"transformer.layers.{layer_idx}.attention.qkv.weight"
                    param = param.t().contiguous()
                elif component == "attn.c_attn.bias":
                    new_name = f"transformer.layers.{layer_idx}.attention.qkv.bias"
                elif component == "attn.c_proj.weight":
                    new_name = f"transformer.layers.{layer_idx}.attention.dense.weight"
                    param = param.t().contiguous()
                elif component == "attn.c_proj.bias":
                    new_name = f"transformer.layers.{layer_idx}.attention.dense.bias"
                elif component == "mlp.c_fc.weight":
                    new_name = f"transformer.layers.{layer_idx}.mlp.fc.weight"
                    param = param.t().contiguous()
                elif component == "mlp.c_fc.bias":
                    new_name = f"transformer.layers.{layer_idx}.mlp.fc.bias"
                elif component == "mlp.c_proj.weight":
                    new_name = f"transformer.layers.{layer_idx}.mlp.proj.weight"
                    param = param.t().contiguous()
                elif component == "mlp.c_proj.bias":
                    new_name = f"transformer.layers.{layer_idx}.mlp.proj.bias"
                else:
                    new_name = name
                    logger.warning(f"Unknown transformer component: {component}")

                trtllm_weights[new_name] = param

            elif name.startswith("transformer.ln_f."):
                # Final layer norm
                if "weight" in name:
                    trtllm_weights["transformer.ln_f.weight"] = param
                else:
                    trtllm_weights["transformer.ln_f.bias"] = param

            elif name.startswith("transformer.wpe."):
                # Position embeddings
                trtllm_weights["transformer.position_embedding.weight"] = param

            elif name.startswith("text_emb."):
                # Text embedding -> vocab embedding
                trtllm_weights["transformer.vocab_embedding.weight"] = param

            elif name.startswith("speech_emb."):
                # Skip - saved separately for custom handling
                continue

            elif name.startswith("speech_head."):
                # Skip - saved separately as speech_head.npz
                # TRT-LLM will use tied weights (share_embedding_table=True)
                continue

            elif name.startswith("spkr_enc."):
                continue

            else:
                logger.warning(f"Skipping unknown weight: {name}")

        return trtllm_weights
