import os
import math
from dataclasses import dataclass
from pathlib import Path

import time
from typing import Generator, Tuple, Optional

import librosa
import numpy as np
import torch
import torch.nn.functional as F

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
import logging
logger = logging.getLogger(__name__)


def setup_cuda_optimizations():
    """Enable CUDA optimizations for faster inference."""
    if torch.cuda.is_available():
        # Enable TF32 for Ampere+ GPUs (including L4)
        # Use new API (PyTorch 2.0+) to avoid deprecation warnings
        torch.set_float32_matmul_precision('high')  # Enables TF32 for matmul
        # Enable cudnn autotuner
        torch.backends.cudnn.benchmark = True
        # Disable debug/profiling overhead
        torch.backends.cudnn.enabled = True

REPO_ID = "ResembleAI/chatterbox-turbo"


def punc_norm(text: str) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("…", ", "),
        (":", ","),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device=None, dtype=None):
        self.t3 = self.t3.to(device=device, dtype=dtype)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                # Only convert floating point tensors to new dtype
                is_fp = v.is_floating_point()
                self.gen[k] = v.to(device=device, dtype=dtype if is_fp else None)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


@dataclass
class StreamingMetrics:
    """Metrics for streaming TTS generation"""
    latency_to_first_chunk: Optional[float] = None
    rtf: Optional[float] = None
    total_generation_time: Optional[float] = None
    total_audio_duration: Optional[float] = None
    chunk_count: int = 0


class ChatterboxTurboTTS:
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
        dtype: torch.dtype = torch.float32,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.dtype = dtype
        self._compiled = False

        # Pre-create logits processors (avoid recreating each call)
        self._logits_processors_cache = {}

    def compile_models(self, mode: str = "default"):
        """
        Compile models for faster inference.

        Args:
            mode: Compilation mode
                - "default": torch.compile with default settings (recommended for streaming)
                - "reduce-overhead": Uses CUDA graphs (NOT recommended for autoregressive)
                - "max-autotune": Maximum optimization (slower compile, faster run)
                - "tensorrt": Use TensorRT backend (requires torch-tensorrt)

        Note: "reduce-overhead" mode uses CUDA graphs which don't work well with
        autoregressive generation. Use "default" mode for streaming TTS.
        """
        if self._compiled:
            logger.warning("Models already compiled, skipping")
            return

        logger.info(f"Compiling models with mode: {mode}")

        if mode == "tensorrt":
            try:
                import torch_tensorrt
                # TensorRT compilation for T3 transformer
                self.t3.tfmr = torch_tensorrt.compile(
                    self.t3.tfmr,
                    inputs=[torch_tensorrt.Input(
                        min_shape=[1, 1, 1024],
                        opt_shape=[1, 512, 1024],
                        max_shape=[1, 2048, 1024],
                        dtype=torch.float16 if self.dtype == torch.float16 else torch.float32,
                    )],
                    enabled_precisions={torch.float16} if self.dtype == torch.float16 else {torch.float32},
                    truncate_long_and_double=True,
                )
                logger.info("T3 compiled with TensorRT")
            except ImportError:
                logger.warning("torch-tensorrt not installed, falling back to torch.compile")
                mode = "default"  # Use default, not reduce-overhead (CUDA graphs issue)
            except Exception as e:
                logger.warning(f"TensorRT compilation failed: {e}, falling back to torch.compile")
                mode = "default"  # Use default, not reduce-overhead (CUDA graphs issue)

        if mode != "tensorrt":
            compile_kwargs = {}
            if mode == "reduce-overhead":
                compile_kwargs = {"mode": "reduce-overhead"}
            elif mode == "max-autotune":
                compile_kwargs = {"mode": "max-autotune"}

            # Compile T3 transformer
            self.t3.tfmr = torch.compile(self.t3.tfmr, **compile_kwargs)
            # Compile speech embedding and head
            self.t3.speech_emb = torch.compile(self.t3.speech_emb, **compile_kwargs)
            self.t3.speech_head = torch.compile(self.t3.speech_head, **compile_kwargs)
            # Compile S3Gen flow model
            self.s3gen.flow = torch.compile(self.s3gen.flow, **compile_kwargs)
            logger.info(f"Models compiled with torch.compile (mode={mode})")

        self._compiled = True

    def to_half(self):
        """Convert models to float16 for faster inference."""
        self.dtype = torch.float16
        self.t3 = self.t3.half()
        self.s3gen = self.s3gen.half()
        if self.conds is not None:
            self.conds = self.conds.to(dtype=torch.float16)
        logger.info("Models converted to float16")
        return self

    def to_bfloat16(self):
        """Convert models to bfloat16 (better for some ops)."""
        self.dtype = torch.bfloat16
        self.t3 = self.t3.to(torch.bfloat16)
        self.s3gen = self.s3gen.to(torch.bfloat16)
        if self.conds is not None:
            self.conds = self.conds.to(dtype=torch.bfloat16)
        logger.info("Models converted to bfloat16")
        return self

    @classmethod
    def from_local(
        cls,
        ckpt_dir,
        device,
        dtype: str = "float32",
        compile_mode: Optional[str] = None,
    ) -> 'ChatterboxTurboTTS':
        """
        Load model from local checkpoint.

        Args:
            ckpt_dir: Path to checkpoint directory
            device: Device to load model on ("cuda", "cpu", "mps")
            dtype: Data type ("float32", "float16", "bfloat16")
            compile_mode: Compilation mode (None, "default", "reduce-overhead", "max-autotune", "tensorrt")
        """
        ckpt_dir = Path(ckpt_dir)

        # Setup CUDA optimizations
        if device == "cuda":
            setup_cuda_optimizations()

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        # Determine torch dtype
        torch_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(dtype, torch.float32)

        ve = VoiceEncoder()
        ve.load_state_dict(
            load_file(ckpt_dir / "ve.safetensors")
        )
        ve.to(device).eval()

        # Turbo specific hp
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
        t3.to(device=device, dtype=torch_dtype).eval()

        s3gen = S3Gen(meanflow=True)
        weights = load_file(ckpt_dir / "s3gen_meanflow.safetensors")
        s3gen.load_state_dict(
            weights, strict=True
        )
        s3gen.to(device=device, dtype=torch_dtype).eval()

        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if len(tokenizer) != 50276:
            print(f"WARNING: Tokenizer len {len(tokenizer)} != 50276")

        conds = None
        builtin_voice = ckpt_dir / "conds.pt"
        if builtin_voice.exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device=device, dtype=torch_dtype)

        instance = cls(t3, s3gen, ve, tokenizer, device, conds=conds, dtype=torch_dtype)

        # Compile if requested
        if compile_mode:
            instance.compile_models(compile_mode)

        return instance

    @classmethod
    def from_pretrained(
        cls,
        device: str,
        dtype: str = "float32",
        compile_mode: Optional[str] = None,
    ) -> 'ChatterboxTurboTTS':
        """
        Load model from HuggingFace Hub.

        Args:
            device: Device to load model on ("cuda", "cpu", "mps")
            dtype: Data type - affects precision and speed
                - "float32": Full precision (default, most accurate)
                - "float16": Half precision (faster, slight quality loss)
                - "bfloat16": Brain float16 (faster, better precision than fp16)
            compile_mode: Optional compilation for faster inference
                - None: No compilation (default)
                - "default": Basic torch.compile
                - "reduce-overhead": Good for streaming (small batches)
                - "max-autotune": Slowest compile, fastest runtime
                - "tensorrt": Use TensorRT (requires torch-tensorrt)

        Example:
            # Fast inference on GPU with compilation
            model = ChatterboxTurboTTS.from_pretrained(
                device="cuda",
                dtype="float16",
                compile_mode="reduce-overhead"
            )
        """
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"

        local_path = snapshot_download(
            repo_id=REPO_ID,
            token=os.getenv("HF_TOKEN") or True,
            # Optional: Filter to download only what you need
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"]
        )

        return cls.from_local(local_path, device, dtype=dtype, compile_mode=compile_mode)

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        assert len(s3gen_ref_wav) / _sr > 5.0, "Audio prompt must be longer than 5 seconds!"

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device, dtype=self.dtype)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict).to(device=self.device, dtype=self.dtype)

    def generate(
        self,
        text,
        repetition_penalty=1.2,
        top_p=0.95,
        audio_prompt_path=None,
        exaggeration=0.0,
        temperature=0.8,
        top_k=1000,
    ):
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"


        # Norm and tokenize text
        text = punc_norm(text)
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

        # Remove OOV tokens and add silence to end
        speech_tokens = speech_tokens[speech_tokens < 6561]
        speech_tokens = speech_tokens.to(self.device)
        silence = torch.tensor([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL]).long().to(self.device)
        speech_tokens = torch.cat([speech_tokens, silence])

        wav, _ = self.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=self.conds.gen,
            n_cfm_timesteps=2,
        )
        wav = wav.squeeze(0).detach().cpu()
        return wav.unsqueeze(0)

    def _process_token_chunk(
        self,
        new_tokens: torch.Tensor,
        all_tokens_so_far: torch.Tensor,
        context_window: int,
        start_time: float,
        metrics: StreamingMetrics,
        fade_duration: float = 0.02,
    ) -> Tuple[Optional[torch.Tensor], float, bool]:
        """Process a chunk of tokens and return audio."""
        # Build tokens_to_process by including context window
        if len(all_tokens_so_far) > 0:
            context_tokens = (
                all_tokens_so_far[-context_window:]
                if len(all_tokens_so_far) > context_window
                else all_tokens_so_far
            )
            tokens_to_process = torch.cat([context_tokens, new_tokens], dim=-1)
            context_length = len(context_tokens)
        else:
            tokens_to_process = new_tokens
            context_length = 0

        # Filter invalid tokens
        tokens_to_process = tokens_to_process[tokens_to_process < 6561]
        if len(tokens_to_process) == 0:
            return None, 0.0, False

        tokens_to_process = tokens_to_process.to(self.device)

        # Run S3Gen inference
        wav, _ = self.s3gen.inference(
            speech_tokens=tokens_to_process,
            ref_dict=self.conds.gen,
            n_cfm_timesteps=2,
        )
        wav = wav.squeeze(0).detach().cpu().numpy()

        # Crop out context portion - only return new audio
        if context_length > 0:
            samples_per_token = len(wav) / len(tokens_to_process)
            skip_samples = int(context_length * samples_per_token)
            audio_chunk = wav[skip_samples:]
        else:
            audio_chunk = wav

        if len(audio_chunk) == 0:
            return None, 0.0, False

        # Apply fade-in to smooth chunk boundaries
        fade_samples = int(fade_duration * self.sr)
        if fade_samples > 0 and fade_samples < len(audio_chunk):
            fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=audio_chunk.dtype)
            audio_chunk[:fade_samples] *= fade_in

        # Compute audio duration
        audio_duration = len(audio_chunk) / self.sr
        audio_tensor = torch.from_numpy(audio_chunk).unsqueeze(0)

        # Update first-chunk latency metric
        if metrics.chunk_count == 0:
            metrics.latency_to_first_chunk = time.time() - start_time

        metrics.chunk_count += 1
        return audio_tensor, audio_duration, True

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
        Streaming TTS generation that yields audio chunks as they are generated.

        Args:
            text: Input text to synthesize
            audio_prompt_path: Optional path to reference audio for voice cloning
            exaggeration: Emotion exaggeration factor (not used in turbo)
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Repetition penalty
            chunk_size: Number of speech tokens per chunk
            context_window: Number of previous tokens to include for audio coherence
            fade_duration: Seconds to apply linear fade-in on each chunk

        Yields:
            Tuple of (audio_chunk, metrics) where audio_chunk is a torch.Tensor
        """
        start_time = time.time()
        metrics = StreamingMetrics()

        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        text_tokens = text_tokens.input_ids.to(self.device)

        total_audio_duration = 0.0
        all_tokens = torch.tensor([], dtype=torch.long, device=self.device)
        chunk_buffer = []

        # Stream tokens from T3
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

                # When we have enough tokens, process and yield audio
                if len(chunk_buffer) >= chunk_size:
                    new_tokens = torch.cat(chunk_buffer, dim=-1).squeeze(0)

                    audio_tensor, audio_duration, success = self._process_token_chunk(
                        new_tokens, all_tokens, context_window, start_time, metrics, fade_duration
                    )

                    if success:
                        total_audio_duration += audio_duration
                        yield audio_tensor, metrics

                    # Update all tokens
                    all_tokens = torch.cat([all_tokens, new_tokens], dim=-1) if len(all_tokens) > 0 else new_tokens
                    chunk_buffer = []

            # Process remaining tokens in buffer
            if chunk_buffer:
                new_tokens = torch.cat(chunk_buffer, dim=-1).squeeze(0)

                audio_tensor, audio_duration, success = self._process_token_chunk(
                    new_tokens, all_tokens, context_window, start_time, metrics, fade_duration
                )

                if success:
                    total_audio_duration += audio_duration
                    yield audio_tensor, metrics

        # Final metrics
        metrics.total_generation_time = time.time() - start_time
        metrics.total_audio_duration = total_audio_duration
        if total_audio_duration > 0:
            metrics.rtf = metrics.total_generation_time / total_audio_duration

    def _get_logits_processors(
        self,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
    ) -> LogitsProcessorList:
        """Get or create cached logits processors."""
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

    def _inference_stream_turbo(
        self,
        t3_cond: T3Cond,
        text_tokens: torch.Tensor,
        temperature: float = 0.8,
        top_k: int = 1000,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        max_gen_len: int = 1000,
    ) -> Generator[torch.Tensor, None, None]:
        """
        Streaming version of inference_turbo that yields tokens one at a time.
        Optimized for minimal overhead per token.
        """
        # Get cached logits processors
        logits_processors = self._get_logits_processors(
            temperature, top_k, top_p, repetition_penalty
        )

        # Prepare initial embeddings
        speech_start_token = self.t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        embeds, _ = self.t3.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_start_token,
            cfg_weight=0.0,
        )

        # Pre-allocate token storage (avoid growing list)
        generated_speech_tokens = torch.empty(
            (1, max_gen_len + 1), dtype=torch.long, device=self.device
        )
        num_generated = 0

        # Initial forward pass (includes full context)
        llm_outputs = self.t3.tfmr(inputs_embeds=embeds, use_cache=True)
        past_key_values = llm_outputs.past_key_values

        # Get first token
        speech_logits = self.t3.speech_head(llm_outputs[0][:, -1:])
        processed_logits = logits_processors(speech_start_token, speech_logits[:, -1, :])
        probs = F.softmax(processed_logits, dim=-1)
        next_speech_token = torch.multinomial(probs, num_samples=1)

        generated_speech_tokens[:, num_generated] = next_speech_token.squeeze()
        num_generated += 1
        yield next_speech_token

        current_speech_token = next_speech_token
        stop_token = self.t3.hp.stop_speech_token

        # Main generation loop - optimized
        for _ in range(max_gen_len):
            # Embed current token
            current_speech_embed = self.t3.speech_emb(current_speech_token)

            # Forward pass with KV cache
            llm_outputs = self.t3.tfmr(
                inputs_embeds=current_speech_embed,
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = llm_outputs.past_key_values

            # Get logits and sample
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
