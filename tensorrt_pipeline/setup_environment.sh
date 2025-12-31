#!/bin/bash
# Setup script for TensorRT-LLM environment
# Run this on your target GPU machine (G6/L4, A10, etc.)

set -e

echo "=========================================="
echo "Chatterbox Turbo TensorRT Setup"
echo "=========================================="

# Check NVIDIA driver
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Please install NVIDIA drivers."
    exit 1
fi

echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Check CUDA version
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | cut -c2-)
echo "CUDA Version: ${CUDA_VERSION:-not found}"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv trt_venv
source trt_venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA
echo ""
echo "Installing PyTorch..."
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install TensorRT
echo ""
echo "Installing TensorRT..."
pip install tensorrt==10.3.0 --extra-index-url https://pypi.nvidia.com

# Install TensorRT-LLM
echo ""
echo "Installing TensorRT-LLM..."
pip install tensorrt-llm==0.12.0 --extra-index-url https://pypi.nvidia.com

# Install other dependencies
echo ""
echo "Installing other dependencies..."
pip install \
    numpy>=1.24.0,<1.26.0 \
    transformers==4.46.3 \
    safetensors==0.5.3 \
    huggingface_hub \
    librosa==0.11.0 \
    onnx \
    onnxruntime-gpu \
    polygraphy \
    colored

# Install the chatterbox package
echo ""
echo "Installing Chatterbox Turbo..."
pip install -e ..

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import tensorrt; print(f'TensorRT: {tensorrt.__version__}')"
python -c "import tensorrt_llm; print(f'TensorRT-LLM: {tensorrt_llm.__version__}')"

echo ""
echo "=========================================="
echo "Setup complete!"
echo ""
echo "To activate the environment:"
echo "  source trt_venv/bin/activate"
echo ""
echo "Next steps:"
echo "  1. python 01_export_models.py"
echo "  2. python 02_build_engines.py"
echo "  3. python tensorrt_demo.py"
echo "=========================================="
