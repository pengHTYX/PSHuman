#!/bin/bash
export WANDB_API_KEY=$KEY # replace $KEY with your wandb key
export CUDA_VISIBLE_DEVICES=0,1,2,3
### CMD

accelerate launch --config_file node_config/gpu.yaml --num_processes 4 \
    train_mvdiffusion_unit_unclip.py \
    --config configs/train-768-6view-onlyscan_face.yaml > log/log_$$.txt 2>&1
