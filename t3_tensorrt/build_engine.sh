#!/bin/bash
#
# Build TensorRT engine from ONNX model
#
# Usage:
#   ./build_engine.sh
#   ./build_engine.sh --fp32  # Use FP32 instead of FP16
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ONNX_FILE="t3_transformer.onnx"
ENGINE_FILE="t3_transformer.engine"

# Parse arguments
FP16="--fp16"
if [[ "$1" == "--fp32" ]]; then
    FP16=""
    ENGINE_FILE="t3_transformer_fp32.engine"
fi

echo "============================================================"
echo "Building TensorRT Engine"
echo "============================================================"
echo ""
echo "Input:  $ONNX_FILE"
echo "Output: $ENGINE_FILE"
echo "Mode:   ${FP16:-FP32}"
echo ""

# Check ONNX file exists
if [[ ! -f "$ONNX_FILE" ]]; then
    echo "ERROR: ONNX file not found: $ONNX_FILE"
    echo "Run: python export_onnx.py first"
    exit 1
fi

# Build engine with trtexec
echo "Running trtexec..."
echo ""

/usr/local/tensorrt/bin/trtexec \
    --onnx="$ONNX_FILE" \
    --saveEngine="$ENGINE_FILE" \
    $FP16 \
    --minShapes=inputs_embeds:1x1x1024 \
    --optShapes=inputs_embeds:1x256x1024 \
    --maxShapes=inputs_embeds:1x2048x1024 \
    --workspace=4096 \
    --verbose

echo ""
echo "============================================================"
echo "Build Complete!"
echo "============================================================"
echo ""
echo "Engine saved to: $ENGINE_FILE"
echo "Size: $(du -h "$ENGINE_FILE" | cut -f1)"
echo ""
echo "Next steps:"
echo "  python test_trt.py  # Run benchmark"
echo ""
