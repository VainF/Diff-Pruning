#!/bin/bash

# Execute the Python script with the provided arguments
python prune_test.py \
--config "church.yml" \
--timesteps "100" \
--eta "0" \
--ni \
--doc "post_training" \
--skip_type "quad" \
--pruning_ratio "0.05" \
--use_ema \
--use_pretrained \
--pruner "$1" \
--save_pruned_model "run/pruned_test/church_ddpm_$1.pth" \
--taylor_batch_size "4"