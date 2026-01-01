"""
Chatterbox Turbo TTS - Streaming Text-to-Speech

A fast, streaming TTS model with voice cloning capabilities.
"""

import os
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from scipy.io import wavfile as scipy_wav
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from .models.t3 import T3
from .models.s3tokenizer import S3_SR
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond
from .models.t3.modules.t3_config import T3Config
from .models.s3gen.const import S3GEN_SIL

logger = logging.getLogger(__name__)

REPO_ID = "ResembleAI/chatterbox-turbo"


# =============================================================================
# Utilities
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

    # Capitalize first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple spaces
    text = " ".join(text.split())

    # Replace uncommon punctuation
    replacements = [
        ("…", ", "), (":", ","), ("—", "-"), ("–", "-"), (" ,", ","),
        (""", "\""), (""", "\""), ("'", "'"), ("'", "'"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)

    # Ensure ending punctuation
    text = text.rstrip(" ")
    if not any(text.endswith(p) for p in {".", "!", "?", "-", ","}):
        text += "."

    return text


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Conditionals:
    """Voice conditioning data for T3 and S3Gen models."""
    t3: T3Cond
    gen: dict

    def to(self, device=None):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        torch.save({"t3": self.t3.__dict__, "gen": self.gen}, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        data = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**data['t3']), data['gen'])


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


# =============================================================================
# Main TTS Class
# =============================================================================

class ChatterboxTurboTTS:
    """
    Chatterbox Turbo Text-to-Speech model.

    Supports both batch and streaming generation with optional voice cloning.
    """

    ENC_COND_LEN = 15 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        self.sr = S3GEN_SR
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self._compiled = False
        self._logits_processors_cache = {}
        self._resamplers = {}

    # -------------------------------------------------------------------------
    # Model Loading
    # -------------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, device: str) -> 'ChatterboxTurboTTS':
        """
        Load model from HuggingFace Hub.

        Args:
            device: Device to load model on ("cuda", "cpu", "mps")

        Example:
            model = ChatterboxTurboTTS.from_pretrained(device="cuda")
        """
        if device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS not available, falling back to CPU")
            device = "cpu"

        local_path = snapshot_download(
            repo_id=REPO_ID,
            token=os.getenv("HF_TOKEN") or True,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"]
        )
        return cls.from_local(local_path, device)

    @classmethod
    def from_local(cls, ckpt_dir, device: str) -> 'ChatterboxTurboTTS':
        """Load model from local checkpoint directory."""
        ckpt_dir = Path(ckpt_dir)

        if device == "cuda":
            setup_cuda_optimizations()

        map_location = torch.device('cpu') if device in ["cpu", "mps"] else None

        # Load voice encoder
        ve = VoiceEncoder()
        ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
        ve.to(device).eval()

        # Load T3 model
        hp = T3Config(text_tokens_dict_size=50276)
        hp.llama_config_name = "GPT2_medium"
        hp.speech_tokens_dict_size = 6563
        hp.input_pos_emb = None
        hp.speech_cond_prompt_len = 375
        hp.use_perceiver_resampler = False
        hp.emotion_adv = False

        t3 = T3(hp)
        t3_state = load_file(ckpt_dir / "t3_turbo_v1.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        del t3.tfmr.wte
        t3.to(device=device).eval()

        # Load S3Gen model
        s3gen = S3Gen(meanflow=True)
        s3gen.load_state_dict(load_file(ckpt_dir / "s3gen_meanflow.safetensors"), strict=True)
        s3gen.to(device=device).eval()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load default voice conditionals
        conds = None
        builtin_voice = ckpt_dir / "conds.pt"
        if builtin_voice.exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device=device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    # -------------------------------------------------------------------------
    # Compilation
    # -------------------------------------------------------------------------

    def compile(self):
        """Compile models with torch.compile for faster inference."""
        if self._compiled:
            logger.warning("Models already compiled")
            return

        self.s3gen.flow = torch.compile(self.s3gen.flow)
        self._compiled = True
        logger.info("S3Gen flow compiled")

    # -------------------------------------------------------------------------
    # Voice Conditioning
    # -------------------------------------------------------------------------

    def _get_resampler(self, src_sr: int, dst_sr: int) -> torchaudio.transforms.Resample:
        """Get or create a cached resampler."""
        key = (src_sr, dst_sr)
        if key not in self._resamplers:
            self._resamplers[key] = torchaudio.transforms.Resample(src_sr, dst_sr).to(self.device)
        return self._resamplers[key]

    def prepare_conditionals(self, wav_fpath: str, exaggeration: float = 0.5):
        """
        Prepare voice conditioning from a reference audio file.

        Args:
            wav_fpath: Path to reference audio (must be >5 seconds, .wav format)
            exaggeration: Emotion exaggeration factor
        """
        # Load with scipy (no FFmpeg dependency)
        orig_sr, wav_np = scipy_wav.read(wav_fpath)

        # Convert to float32 [-1, 1]
        if wav_np.dtype == np.int16:
            wav_np = wav_np.astype(np.float32) / 32768.0
        elif wav_np.dtype == np.int32:
            wav_np = wav_np.astype(np.float32) / 2147483648.0
        elif wav_np.dtype != np.float32:
            wav_np = wav_np.astype(np.float32)

        # Convert to mono if stereo
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=1)

        wav = torch.from_numpy(wav_np).to(self.device)

        # Resample to 24kHz
        if orig_sr != S3GEN_SR:
            wav = self._get_resampler(orig_sr, S3GEN_SR)(wav)

        assert len(wav) / S3GEN_SR > 5.0, "Audio prompt must be longer than 5 seconds!"

        # Resample to 16kHz for encoder
        ref_16k = self._get_resampler(S3GEN_SR, S3_SR)(wav)

        # Get S3Gen reference embeddings
        s3gen_ref = self.s3gen.embed_ref(wav[:self.DEC_COND_LEN], S3GEN_SR, device=self.device)

        # Get T3 conditioning tokens
        t3_tokens, _ = self.s3gen.tokenizer.forward(
            [ref_16k[:self.ENC_COND_LEN]],
            max_len=self.t3.hp.speech_cond_prompt_len
        )
        t3_tokens = torch.atleast_2d(t3_tokens).to(self.device)

        # Get speaker embedding
        ve_embed = torch.from_numpy(
            self.ve.embeds_from_wavs([ref_16k.cpu().numpy()], sample_rate=S3_SR)
        ).mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)

        self.conds = Conditionals(t3_cond, s3gen_ref).to(device=self.device)

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------

    def generate(
        self,
        text: str,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.0,
        temperature: float = 0.8,
        top_k: int = 1000,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
    ) -> torch.Tensor:
        """
        Generate speech from text (non-streaming).

        Args:
            text: Input text to synthesize
            audio_prompt_path: Optional path to reference audio for voice cloning
            exaggeration: Emotion exaggeration factor
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Repetition penalty

        Returns:
            Audio tensor of shape (1, samples)
        """
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Call prepare_conditionals() first or provide audio_prompt_path"

        text = normalize_text(text)
        text_tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        text_tokens = text_tokens.input_ids.to(self.device)

        speech_tokens = self.t3.inference_turbo(
            t3_cond=self.conds.t3,
            text_tokens=text_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # Filter and add silence
        speech_tokens = speech_tokens[speech_tokens < 6561].to(self.device)
        silence = torch.tensor([S3GEN_SIL] * 3, dtype=torch.long, device=self.device)
        speech_tokens = torch.cat([speech_tokens, silence])

        wav, _ = self.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=self.conds.gen,
            n_cfm_timesteps=2,
        )
        return wav.squeeze(0).detach().cpu().unsqueeze(0)

    def generate_stream(
        self,
        text: str,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.0,
        temperature: float = 0.8,
        top_k: int = 1000,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        chunk_size: int = 25,
        context_window: int = 500,
        crossfade_ms: float = 20.0,
        cfm_steps: int = 5,
    ) -> Generator[Tuple[torch.Tensor, StreamingMetrics], None, None]:
        """
        Streaming TTS generation that yields audio chunks as they are generated.

        Args:
            text: Input text to synthesize
            audio_prompt_path: Optional path to reference audio for voice cloning
            exaggeration: Emotion exaggeration factor
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Repetition penalty
            chunk_size: Target tokens per chunk after ramp-up
            context_window: Max context tokens for audio coherence
            crossfade_ms: Crossfade duration in ms for smooth chunk boundaries
            cfm_steps: CFM diffusion steps for vocoder

        Yields:
            Tuple of (audio_chunk, metrics)
        """
        start_time = time.time()
        metrics = StreamingMetrics()

        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Call prepare_conditionals() first or provide audio_prompt_path"

        text = normalize_text(text)
        text_tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        text_tokens = text_tokens.input_ids.to(self.device)

        total_audio_duration = 0.0
        all_tokens = torch.tensor([], dtype=torch.long, device=self.device)
        chunk_buffer = []
        prev_tail: Optional[np.ndarray] = None

        # Dynamic ramp-up schedule: [chunk_size, context_window, cfm_steps]
        ramp_schedule = [
            (4, 0, 1),      # Chunk 0: fast first chunk
            (8, 4, 3),      # Chunk 1
            (16, 12, 5),    # Chunk 2
            (32, 28, 7),    # Chunk 3
            (32, 60, 7),    # Chunk 4
            (32, 125, 7),   # Chunk 5
            (32, 200, 7),   # Chunk 6+
        ]

        with torch.inference_mode():
            for token in self._stream_tokens(
                self.conds.t3, text_tokens, temperature, top_k, top_p, repetition_penalty
            ):
                chunk_buffer.append(token)

                schedule_idx = min(metrics.chunk_count, len(ramp_schedule) - 1)
                current_chunk_size, current_context, current_cfm = ramp_schedule[schedule_idx]

                if len(chunk_buffer) >= current_chunk_size:
                    new_tokens = torch.cat(chunk_buffer, dim=-1).squeeze(0)

                    audio_tensor, audio_duration, success, prev_tail = self._process_chunk(
                        new_tokens, all_tokens, current_context, start_time, metrics,
                        prev_tail, crossfade_ms, current_cfm
                    )

                    if success:
                        total_audio_duration += audio_duration
                        yield audio_tensor, metrics

                    all_tokens = torch.cat([all_tokens, new_tokens], dim=-1) if len(all_tokens) > 0 else new_tokens
                    chunk_buffer = []

            # Process remaining tokens
            if chunk_buffer:
                new_tokens = torch.cat(chunk_buffer, dim=-1).squeeze(0)
                schedule_idx = min(metrics.chunk_count, len(ramp_schedule) - 1)
                _, current_context, current_cfm = ramp_schedule[schedule_idx]

                audio_tensor, audio_duration, success, prev_tail = self._process_chunk(
                    new_tokens, all_tokens, current_context, start_time, metrics,
                    prev_tail, crossfade_ms, current_cfm
                )

                if success:
                    total_audio_duration += audio_duration
                    yield audio_tensor, metrics

            # Flush held-back tail
            if prev_tail is not None and len(prev_tail) > 0:
                final_audio = torch.from_numpy(prev_tail.copy()).unsqueeze(0)
                total_audio_duration += len(prev_tail) / self.sr
                metrics.chunk_count += 1
                yield final_audio, metrics

        metrics.total_generation_time = time.time() - start_time
        metrics.total_audio_duration = total_audio_duration
        if total_audio_duration > 0:
            metrics.rtf = metrics.total_generation_time / total_audio_duration

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _process_chunk(
        self,
        new_tokens: torch.Tensor,
        all_tokens: torch.Tensor,
        context_window: int,
        start_time: float,
        metrics: StreamingMetrics,
        prev_tail: Optional[np.ndarray],
        crossfade_ms: float,
        cfm_steps: int,
    ) -> Tuple[Optional[torch.Tensor], float, bool, Optional[np.ndarray]]:
        """Process a chunk of tokens and return audio with crossfade."""
        chunk_start = time.time()

        # Build tokens with context
        if len(all_tokens) > 0:
            context_tokens = all_tokens[-context_window:] if len(all_tokens) > context_window else all_tokens
            tokens = torch.cat([context_tokens, new_tokens], dim=-1)
            context_length = len(context_tokens)
        else:
            tokens = new_tokens
            context_length = 0

        # Filter invalid tokens
        tokens = tokens[tokens < 6561]
        if len(tokens) == 0:
            return None, 0.0, False, prev_tail

        # Generate audio
        wav, _ = self.s3gen.inference(
            speech_tokens=tokens.to(self.device),
            ref_dict=self.conds.gen,
            n_cfm_timesteps=cfm_steps,
        )
        wav = wav.squeeze(0).detach().cpu().numpy()

        # Crop context portion
        if context_length > 0:
            samples_per_token = len(wav) / len(tokens)
            skip = int(context_length * samples_per_token)

            # Find zero-crossing for clean cut
            search_range = min(100, len(wav) - skip - 1)
            if search_range > 10:
                region = wav[skip:skip + search_range]
                crossings = np.where(np.diff(np.signbit(region)))[0]
                if len(crossings) > 0:
                    skip += crossings[0]

            audio = wav[skip:]
        else:
            audio = wav

        if len(audio) == 0:
            return None, 0.0, False, prev_tail

        # Crossfade processing
        crossfade_samples = int(crossfade_ms * self.sr / 1000)
        audio = audio - np.mean(audio)  # Remove DC offset

        if prev_tail is not None and crossfade_samples > 0:
            blend_len = min(crossfade_samples, len(prev_tail), len(audio))
            if blend_len > 0:
                # Longer blend for low-energy regions
                tail_energy = np.mean(np.abs(prev_tail))
                head_energy = np.mean(np.abs(audio[:blend_len]))
                if tail_energy < 0.02 or head_energy < 0.02:
                    blend_len = min(blend_len * 4, len(prev_tail), len(audio))

                # Equal-power crossfade
                t = np.linspace(0, np.pi / 2, blend_len, dtype=audio.dtype)
                blended = prev_tail[:blend_len] * np.cos(t) + audio[:blend_len] * np.sin(t)
                audio = np.concatenate([blended, audio[blend_len:]])

        # Hold back tail for next chunk
        if len(audio) > crossfade_samples:
            new_tail = audio[-crossfade_samples:].copy()
            audio = audio[:-crossfade_samples]
        else:
            new_tail = audio.copy()
            audio = np.array([], dtype=audio.dtype)

        audio_duration = len(audio) / self.sr
        audio_tensor = torch.from_numpy(audio.copy()).unsqueeze(0)

        # Update metrics
        chunk_time = time.time() - chunk_start
        if metrics.chunk_count == 0:
            metrics.latency_to_first_chunk = time.time() - start_time

        metrics.cumulative_gen_time += chunk_time
        metrics.cumulative_audio_duration += audio_duration
        metrics.buffer_ahead = metrics.cumulative_audio_duration - (time.time() - start_time)
        metrics.chunk_count += 1

        return audio_tensor, audio_duration, True, new_tail

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

    def _stream_tokens(
        self,
        t3_cond: T3Cond,
        text_tokens: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        max_len: int = 1000,
    ) -> Generator[torch.Tensor, None, None]:
        """Stream speech tokens one at a time."""
        processors = self._get_logits_processors(temperature, top_k, top_p, repetition_penalty)

        # Initial embeddings
        start_token = self.t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        embeds, _ = self.t3.prepare_input_embeds(
            t3_cond=t3_cond, text_tokens=text_tokens, speech_tokens=start_token, cfg_weight=0.0
        )

        # Token storage
        generated = torch.empty((1, max_len + 1), dtype=torch.long, device=self.device)
        count = 0

        # Initial forward pass
        outputs = self.t3.tfmr(inputs_embeds=embeds, use_cache=True)
        kv_cache = outputs.past_key_values

        # First token
        logits = self.t3.speech_head(outputs[0][:, -1:])
        logits = processors(start_token, logits[:, -1, :])
        token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        generated[:, count] = token.squeeze()
        count += 1
        yield token

        stop_token = self.t3.hp.stop_speech_token

        # Generation loop
        for _ in range(max_len):
            embed = self.t3.speech_emb(token)
            outputs = self.t3.tfmr(inputs_embeds=embed, past_key_values=kv_cache, use_cache=True)
            kv_cache = outputs.past_key_values

            logits = self.t3.speech_head(outputs[0])
            logits = processors(generated[:, :count], logits[:, -1, :])

            if torch.all(logits == -float("inf")):
                break

            token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            if token.item() == stop_token:
                break

            generated[:, count] = token.squeeze()
            count += 1
            yield token
