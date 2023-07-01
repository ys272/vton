#!/bin/bash
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CUDA_ROOT=/usr/lib/cuda
export SCHP_DIR="$1"
export SCHP_MODEL_TYPE="$2"
export SCHP_INPUT_DIR="$3"
export SCHP_OUTPUT_DIR="$4"
python "$SCHP_DIR/simple_extractor.py" --dataset "$SCHP_MODEL_TYPE" --model-restore "$SCHP_DIR/checkpoints/$SCHP_MODEL_TYPE.pth" --input-dir "$SCHP_INPUT_DIR" --output-dir "$SCHP_OUTPUT_DIR" --logits
