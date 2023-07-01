#!/bin/bash
source ~/miniconda3/bin/activate f
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CUDA_ROOT=/usr/lib/cuda
export SCHP_DIR=/home/yoni/Desktop/f/ext-code/Self-Correction-Human-Parsing
export DEMO_DIR=/home/yoni/Desktop/f/demo
# possible model types are "atr", "lip", "pascal"
export SCHP_MODEL_TYPE=pascal
python $SCHP_DIR/simple_extractor.py --dataset $SCHP_MODEL_TYPE --model-restore $SCHP_DIR/checkpoints/$SCHP_MODEL_TYPE.pth --input-dir $DEMO_DIR/inputs --output-dir $DEMO_DIR/outputs --logits
