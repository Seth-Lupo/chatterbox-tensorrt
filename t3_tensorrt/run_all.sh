#!/bin/bash
#
# Complete TensorRT pipeline for T3 transformer
#
# This script:
# 1. Exports T3 transformer to ONNX
# 2. Builds TensorRT engine
# 3. Runs benchmarks comparing PyTorch vs TensorRT
#
# Usage:
#   ./run_all.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "T3 TensorRT Pipeline"
echo "============================================================"
echo ""

# Step 1: Export to ONNX
echo "Step 1: Exporting to ONNX"
echo "------------------------------------------------------------"
python3 export_onnx.py

echo ""
echo "Step 2: Building TensorRT Engine"
echo "------------------------------------------------------------"
./build_engine.sh

echo ""
echo "Step 3: Running Benchmarks"
echo "------------------------------------------------------------"
python3 test_trt.py

echo ""
echo "============================================================"
echo "Pipeline Complete!"
echo "============================================================"
