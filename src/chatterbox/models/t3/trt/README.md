# T3 TensorRT-LLM Export Module

This module provides utilities for exporting Chatterbox Turbo (T3) models to TensorRT-LLM format for optimized inference.

## Overview

The T3 model architecture was originally designed with runtime voice conditioning via `inputs_embeds`, which is incompatible with TensorRT-LLM. This module performs a **surgical refactor** that:

1. **Bakes voice conditioning at compile time** - One voice per TensorRT engine
2. **Converts to input_ids-only architecture** - Compatible with TensorRT-LLM
3. **Preserves all pretrained weights** - No retraining required
4. **Maintains audio quality** - Exact weight preservation

## Architecture Comparison

### Original T3 Architecture

```
[voice_encoder] → [conditioning_encoder] → [concat with text_emb + speech_emb]
                                                        ↓
                                              [inputs_embeds to transformer]
                                                        ↓
                                              [speech_head → logits]
```

**Problem**: TensorRT-LLM does NOT allow runtime `inputs_embeds`.

### T3ForTRT Architecture (This Module)

```
[FIXED_VOICE_PREFIX from prompt_table] → [prepend to sequence]
                                                    ↓
                    [input_ids] → [unified_embedding_table]
                                                    ↓
                                        [transformer backbone]
                                                    ↓
                                        [speech_head → logits]
```

**Solution**: Voice conditioning is baked into a **prompt table** at compile time. Text and speech tokens use `input_ids` through a unified embedding table.

## Quick Start

### Step 1: Extract Voice Conditioning

Extract the conditioning embeddings for your target voice:

```python
from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.t3.trt import extract_and_save_voice_prefix

# Load your trained T3 model
t3_model = T3.from_pretrained("path/to/checkpoint")
t3_model.eval()

# Prepare voice conditioning data
speaker_emb = voice_encoder(reference_audio)  # 256-dim
voice_prompt_tokens = tokenize_voice_prompt(reference_audio)  # Speech tokens

# Extract and save voice prefix
voice_prefix, metadata = extract_and_save_voice_prefix(
    t3_model=t3_model,
    speaker_emb=speaker_emb,
    output_path="voice_prefix.pt",
    cond_prompt_speech_tokens=voice_prompt_tokens,
    emotion_adv=0.5,
)

print(f"Extracted voice prefix: shape={voice_prefix.shape}")
# Expected: torch.Size([1, 34, 1024])
```

### Step 2: Convert Model Weights

Convert T3 weights to T3ForTRT format:

```python
from chatterbox.models.t3.trt import convert_checkpoint

convert_checkpoint(
    input_path="t3_checkpoint.pth",
    output_path="t3_for_trt_checkpoint.pth",
    voice_prefix_path="voice_prefix.pt",  # Include voice in checkpoint
)
```

### Step 3: Create T3ForTRT Model

```python
from chatterbox.models.t3.trt import create_t3_for_trt, load_converted_checkpoint

# Create model
model = create_t3_for_trt(voice_prefix_path="voice_prefix.pt")

# Load converted weights
model, info = load_converted_checkpoint("t3_for_trt_checkpoint.pth", model)

# Verify voice prefix is set
print(f"Voice prefix: {model.voice_prefix.shape}")
```

### Step 4: Test Generation (PyTorch)

```python
import torch

# Tokenize text
text_tokens = tokenizer.encode("Hello, world!")
text_tokens = torch.tensor([text_tokens])

# Generate speech tokens
speech_tokens = model.generate(
    text_ids=text_tokens,
    max_new_tokens=1000,
    temperature=0.8,
    top_p=0.95,
)

print(f"Generated {speech_tokens.shape[1]} speech tokens")
```

### Step 5: Export to TensorRT-LLM

```python
from chatterbox.models.t3.trt import full_export_pipeline

paths = full_export_pipeline(
    t3_checkpoint_path="t3_checkpoint.pth",
    voice_prefix_path="voice_prefix.pt",
    output_dir="trt_export/",
)

print(f"Checkpoint: {paths['checkpoint_dir']}")
print(f"Build script: {paths['build_script']}")
```

### Step 6: Build TensorRT Engine

```bash
# Run the generated build script
bash trt_export/build_engine.sh
```

Or manually with TensorRT-LLM:

```bash
trtllm-build \
    --checkpoint_dir trt_export/ \
    --output_dir trt_export/engine \
    --gemm_plugin float16 \
    --gpt_attention_plugin float16 \
    --max_input_len 2048 \
    --max_seq_len 6144 \
    --max_batch_size 1 \
    --use_paged_context_fmha enable \
    --use_prompt_tuning enable \
    --max_prompt_embedding_table_size 64
```

## File Structure

```
trt/
├── __init__.py              # Module exports
├── README.md                # This documentation
├── t3_for_trt.py           # T3ForTRT model class
├── extract_conditioning.py  # Voice prefix extraction
├── convert_weights.py       # Weight conversion utilities
├── export_trtllm.py        # TensorRT-LLM export
└── validate.py             # Validation utilities
```

## Token Space

The T3ForTRT model uses a **unified token space**:

| Token Range | Token Type | Original Space |
|-------------|------------|----------------|
| `[0, 704)` | Text tokens | `text_vocab` |
| `[704, 8898)` | Speech tokens | `speech_vocab` |

For multilingual models:
| Token Range | Token Type | Original Space |
|-------------|------------|----------------|
| `[0, 2454)` | Text tokens | `text_vocab` |
| `[2454, 10648)` | Speech tokens | `speech_vocab` |

## Sequence Layout

During inference, the sequence is structured as:

```
[VOICE_PREFIX (34 tokens)] [TEXT_TOKENS (variable)] [SPEECH_TOKENS (autoregressive)]
     └── Baked at compile time    └── Dynamic input       └── Generated one-by-one
```

## Prompt Table (Voice Conditioning)

TensorRT-LLM's **prompt tuning** mechanism is used to bake voice conditioning:

- **Prompt ID**: 0 (single voice per engine)
- **Prompt Length**: 34 tokens (typical)
  - 1 speaker embedding projection
  - 32 perceiver-resampled voice prompt tokens
  - 1 emotion adversarial token
- **Prompt Embedding**: Shape `(34, 1024)` in float16

At runtime, passing `prompt_id=0` prepends the baked voice embeddings.

## Validation

Validate your export with the included test suite:

```python
from chatterbox.models.t3.trt import run_all_tests, print_validation_report

results = run_all_tests(
    t3_model=original_model,
    t3_for_trt_model=converted_model,
    test_text_tokens=text_tokens,
    test_t3_cond=conditioning_data,
    checkpoint_dir="trt_export/",
)

print_validation_report(results)
```

Or via CLI:

```bash
python -m chatterbox.models.t3.trt.validate \
    --checkpoint_dir trt_export/ \
    --prompt_table trt_export/prompt_table.npz
```

## API Reference

### T3ForTRT

```python
class T3ForTRT(nn.Module):
    """TensorRT-LLM compatible T3 model."""

    def __init__(self, config: T3ForTRTConfig, voice_prefix: Optional[Tensor] = None):
        ...

    def set_voice_prefix(self, voice_prefix: Tensor):
        """Set the baked voice conditioning."""
        ...

    def forward(
        self,
        input_ids: Tensor,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = True,
        prepend_voice_prefix: bool = True,
    ) -> Dict[str, Tensor]:
        """Forward pass compatible with TensorRT-LLM."""
        ...

    def generate(
        self,
        text_ids: Tensor,
        max_new_tokens: int = 1000,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> Tensor:
        """Generate speech tokens autoregressively."""
        ...
```

### T3ForTRTConfig

```python
@dataclass
class T3ForTRTConfig:
    text_vocab_size: int = 704
    speech_vocab_size: int = 8194
    hidden_size: int = 1024
    voice_prefix_len: int = 34
    max_text_tokens: int = 2048
    max_speech_tokens: int = 4096
    llama_config_name: str = "Llama_520M"

    @property
    def total_vocab_size(self) -> int:
        return self.text_vocab_size + self.speech_vocab_size

    @classmethod
    def from_t3_config(cls, t3_config: T3Config, voice_prefix_len: int = 34):
        ...
```

## Limitations

1. **One Voice Per Engine**: The voice is baked at compile time. To support multiple voices, build separate engines or use larger prompt tables.

2. **No CFG (Classifier-Free Guidance)**: The current implementation doesn't support CFG during TensorRT inference. Add separate engines for unconditional generation if needed.

3. **Positional Embeddings**: The learned positional embeddings are handled separately from the unified embedding table. TensorRT-LLM may require custom plugins for optimal performance.

## Troubleshooting

### "Voice prefix not set"

Ensure you call `set_voice_prefix()` or pass `voice_prefix_path` when creating the model:

```python
model = create_t3_for_trt(voice_prefix_path="voice_prefix.pt")
```

### "Shape mismatch in weights"

The weight conversion may fail if the source checkpoint has a different architecture. Ensure:
- Same `llama_config_name` (e.g., "Llama_520M")
- Same text/speech vocab sizes
- Same hidden size (1024)

### TensorRT-LLM build errors

Check that:
1. TensorRT-LLM is installed correctly
2. CUDA version is compatible
3. `max_prompt_embedding_table_size` >= voice prefix length

## License

MIT License - See main repository for details.
