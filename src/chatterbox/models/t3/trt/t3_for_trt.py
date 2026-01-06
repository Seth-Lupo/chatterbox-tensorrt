# Copyright (c) 2025 Resemble AI
# MIT License
"""
T3ForTRT: TensorRT-LLM Compatible T3 Model

This module provides a surgically refactored version of the T3 model that is
compatible with TensorRT-LLM export. Key changes from the original T3:

1. NO runtime inputs_embeds - only input_ids are accepted
2. Voice conditioning is BAKED at compile time via prompt table
3. Unified token space: text tokens [0, text_vocab) + speech tokens [text_vocab, total_vocab)
4. Standard decoder-only transformer architecture

Token Space Layout:
    [0, text_vocab_size)                              -> Text tokens
    [text_vocab_size, text_vocab_size + speech_vocab) -> Speech tokens

Sequence Layout During Inference:
    [FIXED_VOICE_PREFIX] [TEXT_TOKENS] [SPEECH_TOKENS (autoregressive)]

    Where:
    - FIXED_VOICE_PREFIX: Baked conditioning from prompt table (34 tokens typically)
    - TEXT_TOKENS: Dynamic text input encoded as input_ids
    - SPEECH_TOKENS: Autoregressively generated, one token at a time
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import LlamaModel, LlamaConfig, GPT2Config, GPT2Model

from ..modules.learned_pos_emb import LearnedPositionEmbeddings
from ..modules.t3_config import T3Config
from ..llama_configs import LLAMA_CONFIGS

logger = logging.getLogger(__name__)


@dataclass
class T3ForTRTConfig:
    """Configuration for T3ForTRT model."""

    # Token vocabulary sizes
    text_vocab_size: int = 704
    speech_vocab_size: int = 8194

    # Special tokens (in unified space)
    start_text_token: int = 255
    stop_text_token: int = 0
    start_speech_token: int = 6561  # Will be remapped to text_vocab + 6561
    stop_speech_token: int = 6562  # Will be remapped to text_vocab + 6562

    # Model architecture
    hidden_size: int = 1024
    llama_config_name: str = "Llama_520M"

    # Sequence lengths
    max_text_tokens: int = 2048
    max_speech_tokens: int = 4096

    # Voice prefix configuration (set after extraction)
    voice_prefix_len: int = 34  # Default: 1 speaker + 32 perceiver + 1 emotion

    @property
    def total_vocab_size(self) -> int:
        """Total vocabulary size (text + speech)."""
        return self.text_vocab_size + self.speech_vocab_size

    @property
    def unified_start_speech_token(self) -> int:
        """Start speech token in unified token space."""
        return self.text_vocab_size + self.start_speech_token

    @property
    def unified_stop_speech_token(self) -> int:
        """Stop speech token in unified token space."""
        return self.text_vocab_size + self.stop_speech_token

    @classmethod
    def from_t3_config(cls, t3_config: T3Config, voice_prefix_len: int = 34):
        """Create T3ForTRTConfig from original T3Config."""
        return cls(
            text_vocab_size=t3_config.text_tokens_dict_size,
            speech_vocab_size=t3_config.speech_tokens_dict_size,
            start_text_token=t3_config.start_text_token,
            stop_text_token=t3_config.stop_text_token,
            start_speech_token=t3_config.start_speech_token,
            stop_speech_token=t3_config.stop_speech_token,
            llama_config_name=t3_config.llama_config_name,
            max_text_tokens=t3_config.max_text_tokens,
            max_speech_tokens=t3_config.max_speech_tokens,
            voice_prefix_len=voice_prefix_len,
        )


class UnifiedEmbedding(nn.Module):
    """
    Unified embedding layer for text and speech tokens.

    This module handles the dual embedding spaces of T3:
    - Text tokens: indices [0, text_vocab_size) use text_emb + text_pos_emb
    - Speech tokens: indices [text_vocab_size, total_vocab) use speech_emb + speech_pos_emb

    The positional embeddings are computed based on the cumulative count of each
    token type, allowing text and speech to have independent position spaces.
    """

    def __init__(self, config: T3ForTRTConfig):
        super().__init__()
        self.config = config

        # Separate embedding tables (matching original T3)
        self.text_emb = nn.Embedding(config.text_vocab_size, config.hidden_size)
        self.speech_emb = nn.Embedding(config.speech_vocab_size, config.hidden_size)

        # Learned positional embeddings (matching original T3)
        max_text_seq_len = config.max_text_tokens + 2
        max_speech_seq_len = config.max_speech_tokens + 4

        self.text_pos_emb = LearnedPositionEmbeddings(max_text_seq_len, config.hidden_size)
        self.speech_pos_emb = LearnedPositionEmbeddings(max_speech_seq_len, config.hidden_size)

    def forward(
        self,
        input_ids: Tensor,
        text_positions: Optional[Tensor] = None,
        speech_positions: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute embeddings for input_ids with appropriate positional encoding.

        Args:
            input_ids: Token IDs in unified space, shape (B, seq_len)
            text_positions: Optional explicit text positions, shape (B, seq_len)
            speech_positions: Optional explicit speech positions, shape (B, seq_len)

        Returns:
            embeddings: Token + positional embeddings, shape (B, seq_len, hidden_size)
        """
        B, seq_len = input_ids.shape
        device = input_ids.device

        # Determine which tokens are text vs speech
        is_text = input_ids < self.config.text_vocab_size
        is_speech = ~is_text

        # Initialize output embeddings
        embeddings = torch.zeros(
            B, seq_len, self.config.hidden_size,
            device=device, dtype=self.text_emb.weight.dtype
        )

        # Process text tokens
        if is_text.any():
            text_ids = input_ids.clone()
            text_ids[is_speech] = 0  # Placeholder for speech positions
            text_token_emb = self.text_emb(text_ids)

            # Compute text positions if not provided
            if text_positions is None:
                # Cumulative count of text tokens per position
                text_positions = (is_text.cumsum(dim=-1) - 1).clamp(min=0)

            text_pos_emb = self.text_pos_emb.emb(text_positions)
            text_full_emb = text_token_emb + text_pos_emb

            # Only keep text embeddings where we have text tokens
            embeddings = torch.where(
                is_text.unsqueeze(-1).expand_as(embeddings),
                text_full_emb,
                embeddings
            )

        # Process speech tokens
        if is_speech.any():
            # Convert to speech-local indices
            speech_ids = (input_ids - self.config.text_vocab_size).clamp(min=0)
            speech_ids[is_text] = 0  # Placeholder for text positions
            speech_token_emb = self.speech_emb(speech_ids)

            # Compute speech positions if not provided
            if speech_positions is None:
                # Cumulative count of speech tokens per position
                speech_positions = (is_speech.cumsum(dim=-1) - 1).clamp(min=0)

            speech_pos_emb = self.speech_pos_emb.emb(speech_positions)
            speech_full_emb = speech_token_emb + speech_pos_emb

            # Only keep speech embeddings where we have speech tokens
            embeddings = torch.where(
                is_speech.unsqueeze(-1).expand_as(embeddings),
                speech_full_emb,
                embeddings
            )

        return embeddings

    def embed_text_only(self, text_ids: Tensor) -> Tensor:
        """
        Embed text tokens only (for initial context processing).

        Args:
            text_ids: Text token IDs (NOT in unified space), shape (B, seq_len)

        Returns:
            embeddings: Text embeddings with positional encoding
        """
        text_emb = self.text_emb(text_ids)
        text_pos_emb = self.text_pos_emb(text_ids)
        return text_emb + text_pos_emb

    def embed_speech_only(self, speech_ids: Tensor, position_offset: int = 0) -> Tensor:
        """
        Embed speech tokens only (for autoregressive generation).

        Args:
            speech_ids: Speech token IDs (NOT in unified space), shape (B, seq_len)
            position_offset: Starting position for positional embedding

        Returns:
            embeddings: Speech embeddings with positional encoding
        """
        speech_emb = self.speech_emb(speech_ids)
        # Get positional embeddings for specific positions
        positions = torch.arange(
            position_offset,
            position_offset + speech_ids.size(1),
            device=speech_ids.device
        )
        speech_pos_emb = self.speech_pos_emb.emb(positions)
        return speech_emb + speech_pos_emb

    def get_speech_embedding_at_position(self, speech_id: Tensor, position: int) -> Tensor:
        """
        Get embedding for a single speech token at a specific position.

        Args:
            speech_id: Single speech token ID (NOT in unified space), shape (B, 1)
            position: Position index for positional embedding

        Returns:
            embedding: Speech embedding with positional encoding, shape (B, 1, hidden_size)
        """
        speech_emb = self.speech_emb(speech_id)
        speech_pos_emb = self.speech_pos_emb.get_fixed_embedding(position)
        return speech_emb + speech_pos_emb


class T3ForTRT(nn.Module):
    """
    TensorRT-LLM compatible T3 model.

    This model is designed for export to TensorRT-LLM with the following properties:
    - Accepts only input_ids (no inputs_embeds at runtime)
    - Voice conditioning is baked at compile time via prompt table
    - Uses a unified token space for text and speech
    - Standard decoder-only autoregressive architecture

    The forward pass prepends a fixed voice prefix (from prompt table) to the
    input embeddings, then runs through the transformer backbone.
    """

    def __init__(
        self,
        config: T3ForTRTConfig,
        voice_prefix: Optional[Tensor] = None,
    ):
        """
        Initialize T3ForTRT model.

        Args:
            config: Model configuration
            voice_prefix: Pre-extracted voice conditioning embeddings, shape (1, P, hidden_size)
                         If None, must be set before inference via set_voice_prefix()
        """
        super().__init__()
        self.config = config

        # Initialize transformer backbone
        llama_config_dict = LLAMA_CONFIGS[config.llama_config_name]
        self.is_gpt = llama_config_dict.get("model_type") == "gpt2"

        if self.is_gpt:
            self.transformer_config = GPT2Config(**llama_config_dict)
            self.transformer = GPT2Model(self.transformer_config)
        else:
            self.transformer_config = LlamaConfig(**llama_config_dict)
            self.transformer = LlamaModel(self.transformer_config)

        # Unified embedding layer
        self.embedding = UnifiedEmbedding(config)

        # Output projection head (speech tokens only for generation)
        self.speech_head = nn.Linear(
            config.hidden_size,
            config.speech_vocab_size,
            bias=self.is_gpt
        )

        # Voice prefix (baked conditioning)
        self._voice_prefix: Optional[Tensor] = None
        if voice_prefix is not None:
            self.set_voice_prefix(voice_prefix)

    @property
    def device(self) -> torch.device:
        return self.speech_head.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.speech_head.weight.dtype

    def set_voice_prefix(self, voice_prefix: Tensor):
        """
        Set the voice prefix (baked conditioning embeddings).

        This should be called once with the extracted conditioning embeddings
        for the target voice. The prefix is frozen and will be prepended to
        all input sequences.

        Args:
            voice_prefix: Conditioning embeddings, shape (1, P, hidden_size)
        """
        if voice_prefix.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {voice_prefix.dim()}D")
        if voice_prefix.size(0) != 1:
            raise ValueError(f"Expected batch size 1, got {voice_prefix.size(0)}")
        if voice_prefix.size(2) != self.config.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.config.hidden_size}, "
                f"got {voice_prefix.size(2)}"
            )

        # Register as buffer (not parameter - frozen)
        self.register_buffer("_voice_prefix", voice_prefix)
        self.config.voice_prefix_len = voice_prefix.size(1)
        logger.info(f"Set voice prefix: shape={voice_prefix.shape}")

    @property
    def voice_prefix(self) -> Tensor:
        """Get the voice prefix tensor."""
        if self._voice_prefix is None:
            raise RuntimeError("Voice prefix not set. Call set_voice_prefix() first.")
        return self._voice_prefix

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = True,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        text_positions: Optional[Tensor] = None,
        speech_positions: Optional[Tensor] = None,
        prepend_voice_prefix: bool = True,
    ):
        """
        Forward pass through the T3ForTRT model.

        For initial context processing:
            - Pass text tokens as input_ids
            - Voice prefix is automatically prepended
            - Returns hidden states and initial KV cache

        For autoregressive generation:
            - Pass single speech token as input_ids
            - Set prepend_voice_prefix=False
            - Pass past_key_values from previous step

        Args:
            input_ids: Token IDs in unified space, shape (B, seq_len)
            inputs_embeds: Pre-computed embeddings (for internal use only)
            past_key_values: KV cache from previous forward pass
            use_cache: Whether to return KV cache
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return dict (always True)
            text_positions: Optional explicit text positions
            speech_positions: Optional explicit speech positions
            prepend_voice_prefix: Whether to prepend voice prefix (True for initial, False for generation)

        Returns:
            Dictionary with:
                - logits: Speech logits, shape (B, seq_len, speech_vocab_size)
                - past_key_values: KV cache for next step
                - hidden_states: Final hidden states (optional)
        """
        # Compute embeddings from input_ids
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided")
            inputs_embeds = self.embedding(
                input_ids,
                text_positions=text_positions,
                speech_positions=speech_positions,
            )

        # Prepend voice prefix for initial context
        if prepend_voice_prefix and past_key_values is None:
            voice_prefix = self.voice_prefix.to(dtype=inputs_embeds.dtype)
            # Expand to batch size
            if voice_prefix.size(0) != inputs_embeds.size(0):
                voice_prefix = voice_prefix.expand(inputs_embeds.size(0), -1, -1)
            inputs_embeds = torch.cat([voice_prefix, inputs_embeds], dim=1)

        # Run through transformer
        transformer_out = self.transformer(
            input_ids=None,  # Always use inputs_embeds
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = transformer_out.last_hidden_state

        # Project to speech logits
        logits = self.speech_head(hidden_states)

        return {
            "logits": logits,
            "past_key_values": transformer_out.past_key_values if use_cache else None,
            "hidden_states": hidden_states if output_hidden_states else None,
            "all_hidden_states": transformer_out.hidden_states if output_hidden_states else None,
        }

    @torch.inference_mode()
    def generate(
        self,
        text_ids: Tensor,
        max_new_tokens: int = 1000,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 0,
        repetition_penalty: float = 1.2,
        stop_on_eos: bool = True,
    ) -> Tensor:
        """
        Generate speech tokens autoregressively.

        This is the main inference method for text-to-speech generation.
        The voice is determined by the baked voice_prefix.

        Args:
            text_ids: Text token IDs (in original text space, not unified)
            max_new_tokens: Maximum speech tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling (0 to disable)
            repetition_penalty: Penalty for repeated tokens
            stop_on_eos: Whether to stop on EOS token

        Returns:
            generated_ids: Generated speech token IDs (in original speech space)
        """
        device = self.device
        B = text_ids.size(0)

        # Initial context: voice prefix + text
        text_embeds = self.embedding.embed_text_only(text_ids.to(device))
        voice_prefix = self.voice_prefix.to(dtype=text_embeds.dtype)
        if voice_prefix.size(0) != B:
            voice_prefix = voice_prefix.expand(B, -1, -1)

        initial_embeds = torch.cat([voice_prefix, text_embeds], dim=1)

        # Initial forward pass (process full context)
        output = self.transformer(
            input_ids=None,
            inputs_embeds=initial_embeds,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = output.past_key_values

        # Get initial logits and sample first speech token
        hidden = output.last_hidden_state[:, -1:, :]
        logits = self.speech_head(hidden).squeeze(1)  # (B, speech_vocab)

        # Apply sampling
        next_token = self._sample_token(
            logits, temperature, top_p, top_k, repetition_penalty, generated_ids=None
        )

        generated_ids = [next_token]
        speech_position = 1  # Position 0 was implicit in the start

        # Autoregressive generation loop
        for _ in range(max_new_tokens - 1):
            # Check for EOS
            if stop_on_eos and (next_token == self.config.stop_speech_token).all():
                break

            # Get embedding for the new token
            next_embed = self.embedding.get_speech_embedding_at_position(
                next_token.unsqueeze(1), speech_position
            )

            # Forward pass with KV cache
            output = self.transformer(
                input_ids=None,
                inputs_embeds=next_embed,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = output.past_key_values

            # Get logits and sample
            hidden = output.last_hidden_state
            logits = self.speech_head(hidden).squeeze(1)

            # Stack all generated IDs for repetition penalty
            all_generated = torch.stack(generated_ids, dim=1)

            next_token = self._sample_token(
                logits, temperature, top_p, top_k, repetition_penalty,
                generated_ids=all_generated
            )

            generated_ids.append(next_token)
            speech_position += 1

        # Stack and return
        return torch.stack(generated_ids, dim=1)

    def _sample_token(
        self,
        logits: Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        generated_ids: Optional[Tensor],
    ) -> Tensor:
        """Sample next token from logits with various sampling strategies."""
        # Apply repetition penalty
        if generated_ids is not None and repetition_penalty != 1.0:
            for i in range(logits.size(0)):
                for token_id in generated_ids[i].unique():
                    if logits[i, token_id] > 0:
                        logits[i, token_id] /= repetition_penalty
                    else:
                        logits[i, token_id] *= repetition_penalty

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Apply top-k
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        # Apply top-p (nucleus sampling)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return next_token


# =============================================================================
# Factory Functions
# =============================================================================

def create_t3_for_trt(
    t3_config: Optional[T3Config] = None,
    voice_prefix_path: Optional[str] = None,
) -> T3ForTRT:
    """
    Factory function to create a T3ForTRT model.

    Args:
        t3_config: Original T3 config (defaults to english_only)
        voice_prefix_path: Path to extracted voice prefix file

    Returns:
        Configured T3ForTRT model
    """
    if t3_config is None:
        t3_config = T3Config.english_only()

    # Load voice prefix if provided
    voice_prefix = None
    voice_prefix_len = 34  # Default

    if voice_prefix_path is not None:
        from .extract_conditioning import load_voice_prefix, validate_voice_prefix
        voice_prefix, metadata = load_voice_prefix(voice_prefix_path)
        validate_voice_prefix(voice_prefix, metadata, expected_hidden_size=1024)
        voice_prefix_len = metadata["len_cond"]

    # Create config
    config = T3ForTRTConfig.from_t3_config(t3_config, voice_prefix_len=voice_prefix_len)

    # Create model
    model = T3ForTRT(config, voice_prefix=voice_prefix)

    return model
