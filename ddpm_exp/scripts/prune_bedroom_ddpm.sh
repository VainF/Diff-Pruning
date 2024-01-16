#!/bin/bash

# Execute the Python script with the provided arguments
python prune.py \
--config "bedroom.yml" \
--timesteps "100" \
--eta "0" \
--ni \
--doc "post_training" \
--skip_type "quad" \
--pruning_ratio "0.3" \
--use_ema \
--use_pretrained \
--pruner "$1" \
--save_pruned_model "run/pruned/bedroom_ddpm_$1_0.3.pth" \
--taylor_batch_size "4"