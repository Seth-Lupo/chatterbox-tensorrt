# TensorRT Pipeline for Chatterbox Turbo

This pipeline converts Chatterbox Turbo to optimized TensorRT engines for maximum inference speed.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TENSORRT PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Text ──▶ [TensorRT-LLM T3] ──▶ Tokens ──▶ [TensorRT S3Gen] ──▶ Audio
│                 │                               │               │
│            GPT-2 engine                   ONNX→TRT engine       │
│            (INT8/FP16)                      (FP16)              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Requirements

- NVIDIA GPU (L4, A10, A100, H100)
- CUDA 12.x
- TensorRT 10.x
- TensorRT-LLM 0.9+
- Python 3.10+

## Quick Start

```bash
# 1. Setup environment (run on target GPU machine)
./setup_environment.sh

# 2. Download and export models
python 01_export_models.py

# 3. Build TensorRT engines
python 02_build_engines.py

# 4. Run optimized inference
python tensorrt_demo.py --text "Hello world"
```

## Step-by-Step

### Step 1: Environment Setup

```bash
# Creates venv with TensorRT-LLM and dependencies
./setup_environment.sh
source trt_venv/bin/activate
```

### Step 2: Export Models

```bash
# Downloads Chatterbox Turbo and exports to ONNX/checkpoint format
python 01_export_models.py
```

This creates:
- `exports/t3_checkpoint/` - T3 model for TensorRT-LLM
- `exports/s3gen_flow.onnx` - S3Gen flow model
- `exports/hifigan.onnx` - HiFiGAN vocoder

### Step 3: Build Engines

```bash
# Builds optimized TensorRT engines
python 02_build_engines.py --dtype float16

# For INT8 quantization (requires calibration data)
python 02_build_engines.py --dtype int8 --calibration_data ./audio_samples/
```

This creates:
- `engines/t3_engine/` - TensorRT-LLM engine for T3
- `engines/s3gen_flow.trt` - TensorRT engine for S3Gen
- `engines/hifigan.trt` - TensorRT engine for HiFiGAN

### Step 4: Run Inference

```bash
python tensorrt_demo.py --text "Hello, this is TensorRT accelerated speech!"
```

## Performance Comparison

| Configuration | First Chunk Latency | RTF |
|--------------|--------------------|----|
| PyTorch FP32 | ~500ms | ~0.8 |
| PyTorch FP16 + compile | ~300ms | ~0.5 |
| TensorRT FP16 | ~150ms | ~0.25 |
| TensorRT INT8 | ~100ms | ~0.15 |

## Files

- `setup_environment.sh` - Environment setup script
- `01_export_models.py` - Export models to ONNX/checkpoint format
- `02_build_engines.py` - Build TensorRT engines
- `tensorrt_inference.py` - TensorRT inference wrapper class
- `tensorrt_demo.py` - Demo script with streaming output

## Troubleshooting

### "TensorRT-LLM not found"
```bash
pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com
```

### "CUDA out of memory during engine build"
Reduce batch size or sequence length in `02_build_engines.py`

### "Engine build fails"
Check CUDA/TensorRT version compatibility. L4 GPUs require TensorRT 10.x+
