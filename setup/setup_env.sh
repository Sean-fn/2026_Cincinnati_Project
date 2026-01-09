#!/bin/bash
# Environment setup for M2F2-Det
# Source this file before running training scripts

# CUDA 11.7 environment
export CUDA_HOME=/usr/local/cuda-11.7
export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH

echo "Environment variables set:"
echo "  CUDA_HOME=$CUDA_HOME"
echo "  nvcc version: $(nvcc --version | grep release)"
