# TensorRT-LLM Integration for T3

This directory contains scripts to integrate T3's GPT-2 transformer with TensorRT-LLM for maximum inference performance.

## Quick Start

```bash
# Run the proof of concept
./run_trtllm_poc.sh
```

This will:
1. Export T3's GPT-2 weights to HuggingFace format
2. Benchmark PyTorch with KV cache
3. Provide instructions for TensorRT-LLM engine building

## Files

| File | Description |
|------|-------------|
| `run_trtllm_poc.sh` | Main runner script - start here |
| `build_trtllm_engine.sh` | Full engine build script |
| `test_trtllm_t3.py` | Python test and benchmark script |

## T3 Architecture

T3 is a modified GPT-2 with custom components:

```
Input Text → text_emb → ┐
                        ├→ GPT-2 Transformer → speech_head → Speech Tokens
Audio Prompt → cond_enc → ┘
```

**Key difference from standard GPT-2**: T3 takes **embeddings** as input, not token IDs. The standard `wte` (token embedding) layer is replaced with:
- `speech_emb`: Embeds speech tokens
- `text_emb`: Embeds text tokens
- `cond_enc`: Encodes audio conditioning

## Integration Approaches

### 1. PyTorch + KV Cache (Current)

Already implemented in `benchmark_trt_kvcache.py`. Provides good performance:
- ~5-15x speedup for autoregressive generation
- No additional setup required

```python
# Use KV cache
output = model.t3.tfmr(
    inputs_embeds=embeds,
    past_key_values=cache,
    use_cache=True,
    return_dict=True,
)
cache = output.past_key_values
```

### 2. TensorRT-LLM Engine

For maximum throughput:

```bash
# Step 1: Export checkpoint (done by run_trtllm_poc.sh)
# Checkpoint saved to: checkpoints/t3_gpt2_hf/

# Step 2: Convert to TensorRT-LLM format
python3 -m tensorrt_llm.commands.convert_checkpoint \
    --model_type gpt2 \
    --model_dir checkpoints/t3_gpt2_hf \
    --output_dir checkpoints/t3_gpt2_trtllm \
    --dtype float16

# Step 3: Build engine
trtllm-build \
    --checkpoint_dir checkpoints/t3_gpt2_trtllm \
    --output_dir engines/t3_gpt2 \
    --gemm_plugin float16 \
    --gpt_attention_plugin float16 \
    --max_batch_size 1 \
    --max_input_len 1024 \
    --max_seq_len 2048
```

### 3. Triton Inference Server

For production deployment with batching and scaling:

```
triton_models/
├── t3_tts_ensemble/          # Orchestrates the pipeline
├── speech_embedding/         # PyTorch: speech_emb + cond_enc
├── t3_transformer/           # TensorRT-LLM engine
├── speech_head/              # PyTorch: logit projection
├── s3gen/                    # PyTorch: flow matching
└── hifigan/                  # PyTorch: vocoder
```

## Performance Comparison

| Approach | First Token Latency | Tokens/sec | Notes |
|----------|---------------------|------------|-------|
| PyTorch (naive) | ~50ms | ~100 | No optimization |
| PyTorch + KV cache | ~5ms | ~500 | Recommended baseline |
| PyTorch + compile | ~4ms | ~600 | Easy win |
| TensorRT-LLM | ~2ms | ~1000+ | Maximum perf |

*Actual numbers depend on GPU and sequence length*

## Requirements

- PyTorch 2.0+
- CUDA 11.8+ or 12.x
- For TensorRT-LLM: `pip install tensorrt-llm`

## Troubleshooting

### TensorRT-LLM not found
```bash
pip install tensorrt-llm
# Or use NVIDIA container:
docker pull nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3
```

### Engine build fails
- Ensure CUDA toolkit matches TensorRT-LLM version
- Check GPU memory (engine build can use 8GB+)
- Try reducing max_seq_len if OOM

### Embeddings input issue
T3 uses embeddings input, but TensorRT-LLM GPT-2 expects token IDs. Solutions:
1. Keep embeddings in PyTorch, only accelerate attention
2. Create passthrough embedding layer in engine
3. Use hybrid approach (see `PyTorchWithTRTLLMAttention` in test script)
