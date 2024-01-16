python ldm_prune.py \
--model_path CompVis/ldm-celebahq-256 \
--save_path run/pruned/ldm_celeba_pruned \
--pruning_ratio 0.5 \
--pruner magnitude \
--device cuda:0 \
--batch_size 4 \