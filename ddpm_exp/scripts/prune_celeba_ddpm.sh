#!/bin/bash

# Execute the Python script with the provided arguments
python prune.py \
--config "celeba.yml" \
--timesteps "100" \
--eta "0" \
--ni \
--doc "post_training" \
--skip_type "quad" \
--pruning_ratio "0.3" \
--use_ema \
--use_pretrained \
--pruner "ours" \
--save_pruned_model "run/pruned_final/celeba_T=$1.pth" \
--taylor_batch_size "64" \
--thr "$1"