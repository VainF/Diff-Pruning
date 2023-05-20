#!/bin/bash

# Execute the Python script with the provided arguments
python prune_test.py \
--config "bedroom.yml" \
--timesteps "100" \
--eta "0" \
--ni \
--doc "post_training" \
--skip_type "quad" \
--pruning_ratio "0.05" \
--use_ema \
--use_pretrained \
--pruner "$1" \
--save_pruned_model "run/pruned_test/bedroom_ddpm_$1.pth" \
--taylor_batch_size "4"