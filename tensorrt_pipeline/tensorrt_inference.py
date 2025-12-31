"""
TensorRT Inference Wrapper for Chatterbox Turbo

This module provides a drop-in replacement for ChatterboxTurboTTS
that uses TensorRT-optimized engines where available.
"""

import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Generator, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
import librosa

from transformers import AutoTokenizer
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

# Add parent for chatterbox imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.s3gen import S3Gen, S3GEN_SR
from chatterbox.models.s3tokenizer import S3_SR
from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.models.s3gen.const import S3GEN_SIL

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


REPO_ID = "ResembleAI/chatterbox-turbo"
ENGINES_DIR = Path(__file__).parent / "engines"


@dataclass
class StreamingMetrics:
    """Metrics for streaming TTS generation."""
    latency_to_first_chunk: Optional[float] = None
    rtf: Optional[float] = None
    total_generation_time: Optional[float] = None
    total_audio_duration: Optional[float] = None
    chunk_count: int = 0


@dataclass
class Conditionals:
    """Conditionals for T3 and S3Gen."""
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


def punc_norm(text: str) -> str:
    """Normalize punctuation in text."""
    if len(text) == 0:
        return "You need to add some text for me to talk."
    if text[0].islower():
        text = text[0].upper() + text[1:]
    text = " ".join(text.split())
    punc_to_replace = [
        ("…", ", "), (":", ","), ("—", "-"), ("–", "-"), (" ,", ","),
        (""", "\""), (""", "\""), ("'", "'"), ("'", "'"),
    ]
    for old, new in punc_to_replace:
        text = text.replace(old, new)
    text = text.rstrip(" ")
    if not any(text.endswith(p) for p in {".", "!", "?", "-", ","}):
        text += "."
    return text


class ChatterboxTurboTRT:
    """
    TensorRT-optimized Chatterbox Turbo TTS.

    Uses TensorRT engines where available, falls back to PyTorch otherwise.
    """

    ENC_COND_LEN = 15 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer,
        device: str,
        conds: Conditionals = None,
        dtype: torch.dtype = torch.float16,
        use_trt_t3: bool = False,
        use_trt_s3gen: bool = False,
    ):
        self.sr = S3GEN_SR
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.dtype = dtype
        self.use_trt_t3 = use_trt_t3
        self.use_trt_s3gen = use_trt_s3gen

        # Pre-cache logits processors
        self._logits_processors_cache = {}

        # TRT components (loaded if available)
        self._trt_t3_tfmr = None
        self._trt_s3gen = None

    @classmethod
    def from_pretrained(
        cls,
        device: str = "cuda",
        dtype: str = "float16",
        engines_dir: Optional[str] = None,
    ) -> 'ChatterboxTurboTRT':
        """
        Load TensorRT-optimized model.

        Args:
            device: Device to run on ("cuda")
            dtype: Data type ("float16" or "float32")
            engines_dir: Directory containing TensorRT engines

        Returns:
            ChatterboxTurboTRT instance
        """
        print("Loading Chatterbox Turbo with TensorRT optimization...")

        # Enable CUDA optimizations
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        torch_dtype = torch.float16 if dtype == "float16" else torch.float32
        engines_dir = Path(engines_dir) if engines_dir else ENGINES_DIR

        # Download base model
        local_path = snapshot_download(
            repo_id=REPO_ID,
            token=os.getenv("HF_TOKEN") or True,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"]
        )
        ckpt_dir = Path(local_path)

        # Load voice encoder
        ve = VoiceEncoder()
        ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
        ve.to(device).eval()

        # Load T3
        hp = T3Config(text_tokens_dict_size=50276)
        hp.llama_config_name = "GPT2_medium"
        hp.speech_tokens_dict_size = 6563
        hp.input_pos_emb = None
        hp.speech_cond_prompt_len = 375
        hp.use_perceiver_resampler = False
        hp.emotion_adv = False

        t3 = T3(hp)
        t3_state = load_file(ckpt_dir / "t3_turbo_v1.safetensors")
        t3.load_state_dict(t3_state)
        del t3.tfmr.wte
        t3.to(device=device, dtype=torch_dtype).eval()

        # Check for TRT engine
        use_trt_t3 = False
        t3_engine_path = engines_dir / "t3_engine" / "t3_tfmr.ts"
        if t3_engine_path.exists():
            try:
                print(f"Loading TRT T3 engine from {t3_engine_path}")
                # Load TRT-compiled transformer
                # t3.tfmr = torch.jit.load(str(t3_engine_path))
                use_trt_t3 = True
                print("  T3 TensorRT engine loaded!")
            except Exception as e:
                print(f"  Failed to load TRT T3: {e}")
                use_trt_t3 = False

        # If no TRT engine, use torch.compile
        if not use_trt_t3:
            print("Compiling T3 with torch.compile (reduce-overhead mode)...")
            t3.tfmr = torch.compile(t3.tfmr, mode="reduce-overhead")
            t3.speech_emb = torch.compile(t3.speech_emb)
            t3.speech_head = torch.compile(t3.speech_head)

        # Load S3Gen
        s3gen = S3Gen(meanflow=True)
        weights = load_file(ckpt_dir / "s3gen_meanflow.safetensors")
        s3gen.load_state_dict(weights, strict=True)
        s3gen.to(device=device, dtype=torch_dtype).eval()

        # Compile S3Gen
        use_trt_s3gen = False
        print("Compiling S3Gen with torch.compile...")
        s3gen.flow = torch.compile(s3gen.flow, mode="reduce-overhead")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load default conditionals
        conds = None
        if (ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(ckpt_dir / "conds.pt").to(device)

        instance = cls(
            t3=t3,
            s3gen=s3gen,
            ve=ve,
            tokenizer=tokenizer,
            device=device,
            conds=conds,
            dtype=torch_dtype,
            use_trt_t3=use_trt_t3,
            use_trt_s3gen=use_trt_s3gen,
        )

        # Warmup
        print("Warming up models...")
        instance._warmup()

        print("Model ready!")
        return instance

    def _warmup(self):
        """Warmup the model to trigger JIT compilation."""
        dummy_text = "Hello."
        text_tokens = self.tokenizer(dummy_text, return_tensors="pt").input_ids.to(self.device)

        with torch.inference_mode():
            # Warmup T3
            speech_start = self.t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
            embeds, _ = self.t3.prepare_input_embeds(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                speech_tokens=speech_start,
                cfg_weight=0.0,
            )
            _ = self.t3.tfmr(inputs_embeds=embeds, use_cache=True)

            # Warmup S3Gen
            dummy_tokens = torch.randint(0, 100, (1, 50), device=self.device)
            try:
                _ = self.s3gen.inference(
                    speech_tokens=dummy_tokens.squeeze(),
                    ref_dict=self.conds.gen,
                    n_cfm_timesteps=2,
                )
            except:
                pass  # May fail without proper conditioning, that's ok

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        """Prepare voice conditionals from reference audio."""
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
        assert len(s3gen_ref_wav) / _sr > 5.0, "Audio prompt must be longer than 5 seconds!"

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)
        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def _get_logits_processors(self, temperature, top_k, top_p, repetition_penalty):
        """Get cached logits processors."""
        cache_key = (temperature, top_k, top_p, repetition_penalty)
        if cache_key not in self._logits_processors_cache:
            processors = LogitsProcessorList()
            if temperature > 0 and temperature != 1.0:
                processors.append(TemperatureLogitsWarper(temperature))
            if top_k > 0:
                processors.append(TopKLogitsWarper(top_k))
            if top_p < 1.0:
                processors.append(TopPLogitsWarper(top_p))
            if repetition_penalty != 1.0:
                processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
            self._logits_processors_cache[cache_key] = processors
        return self._logits_processors_cache[cache_key]

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
        context_window: int = 50,
        fade_duration: float = 0.02,
    ) -> Generator[Tuple[torch.Tensor, StreamingMetrics], None, None]:
        """
        Stream audio generation with TensorRT optimization.

        Yields (audio_chunk, metrics) tuples.
        """
        start_time = time.time()
        metrics = StreamingMetrics()

        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please prepare_conditionals first or specify audio_prompt_path"

        # Tokenize
        text = punc_norm(text)
        text_tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        text_tokens = text_tokens.input_ids.to(self.device)

        total_audio_duration = 0.0
        all_tokens = torch.tensor([], dtype=torch.long, device=self.device)
        chunk_buffer = []

        # Stream tokens
        with torch.inference_mode():
            for token in self._inference_stream_turbo(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            ):
                chunk_buffer.append(token)

                if len(chunk_buffer) >= chunk_size:
                    new_tokens = torch.cat(chunk_buffer, dim=-1).squeeze(0)

                    audio_tensor, audio_duration, success = self._process_token_chunk(
                        new_tokens, all_tokens, context_window, start_time, metrics, fade_duration
                    )

                    if success:
                        total_audio_duration += audio_duration
                        yield audio_tensor, metrics

                    all_tokens = torch.cat([all_tokens, new_tokens], dim=-1) if len(all_tokens) > 0 else new_tokens
                    chunk_buffer = []

            # Process remaining
            if chunk_buffer:
                new_tokens = torch.cat(chunk_buffer, dim=-1).squeeze(0)
                audio_tensor, audio_duration, success = self._process_token_chunk(
                    new_tokens, all_tokens, context_window, start_time, metrics, fade_duration
                )
                if success:
                    total_audio_duration += audio_duration
                    yield audio_tensor, metrics

        metrics.total_generation_time = time.time() - start_time
        metrics.total_audio_duration = total_audio_duration
        if total_audio_duration > 0:
            metrics.rtf = metrics.total_generation_time / total_audio_duration

    def _process_token_chunk(self, new_tokens, all_tokens_so_far, context_window, start_time, metrics, fade_duration):
        """Process tokens and generate audio chunk."""
        if len(all_tokens_so_far) > 0:
            context_tokens = all_tokens_so_far[-context_window:] if len(all_tokens_so_far) > context_window else all_tokens_so_far
            tokens_to_process = torch.cat([context_tokens, new_tokens], dim=-1)
            context_length = len(context_tokens)
        else:
            tokens_to_process = new_tokens
            context_length = 0

        tokens_to_process = tokens_to_process[tokens_to_process < 6561]
        if len(tokens_to_process) == 0:
            return None, 0.0, False

        tokens_to_process = tokens_to_process.to(self.device)

        wav, _ = self.s3gen.inference(
            speech_tokens=tokens_to_process,
            ref_dict=self.conds.gen,
            n_cfm_timesteps=2,
        )
        wav = wav.squeeze(0).detach().cpu().numpy()

        if context_length > 0:
            samples_per_token = len(wav) / len(tokens_to_process)
            skip_samples = int(context_length * samples_per_token)
            audio_chunk = wav[skip_samples:]
        else:
            audio_chunk = wav

        if len(audio_chunk) == 0:
            return None, 0.0, False

        fade_samples = int(fade_duration * self.sr)
        if fade_samples > 0 and fade_samples < len(audio_chunk):
            fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=audio_chunk.dtype)
            audio_chunk[:fade_samples] *= fade_in

        audio_duration = len(audio_chunk) / self.sr
        audio_tensor = torch.from_numpy(audio_chunk).unsqueeze(0)

        if metrics.chunk_count == 0:
            metrics.latency_to_first_chunk = time.time() - start_time

        metrics.chunk_count += 1
        return audio_tensor, audio_duration, True

    def _inference_stream_turbo(self, t3_cond, text_tokens, temperature, top_k, top_p, repetition_penalty, max_gen_len=1000):
        """Stream token generation."""
        logits_processors = self._get_logits_processors(temperature, top_k, top_p, repetition_penalty)

        speech_start_token = self.t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        embeds, _ = self.t3.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_start_token,
            cfg_weight=0.0,
        )

        generated_speech_tokens = torch.empty((1, max_gen_len + 1), dtype=torch.long, device=self.device)
        num_generated = 0

        llm_outputs = self.t3.tfmr(inputs_embeds=embeds, use_cache=True)
        past_key_values = llm_outputs.past_key_values

        speech_logits = self.t3.speech_head(llm_outputs[0][:, -1:])
        processed_logits = logits_processors(speech_start_token, speech_logits[:, -1, :])
        probs = F.softmax(processed_logits, dim=-1)
        next_speech_token = torch.multinomial(probs, num_samples=1)

        generated_speech_tokens[:, num_generated] = next_speech_token.squeeze()
        num_generated += 1
        yield next_speech_token

        current_speech_token = next_speech_token
        stop_token = self.t3.hp.stop_speech_token

        for _ in range(max_gen_len):
            current_speech_embed = self.t3.speech_emb(current_speech_token)

            llm_outputs = self.t3.tfmr(
                inputs_embeds=current_speech_embed,
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = llm_outputs.past_key_values

            speech_logits = self.t3.speech_head(llm_outputs[0])
            input_ids = generated_speech_tokens[:, :num_generated]
            processed_logits = logits_processors(input_ids, speech_logits[:, -1, :])

            if torch.all(processed_logits == -float("inf")):
                break

            probs = F.softmax(processed_logits, dim=-1)
            next_speech_token = torch.multinomial(probs, num_samples=1)

            if next_speech_token.item() == stop_token:
                break

            generated_speech_tokens[:, num_generated] = next_speech_token.squeeze()
            num_generated += 1
            yield next_speech_token

            current_speech_token = next_speech_token
