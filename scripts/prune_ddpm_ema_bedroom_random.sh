python ddpm_prune.py \
--dataset data/lsun/bedroom \
--model_path google/ddpm-ema-bedroom-256 \
--save_path run/pruned/ddpm_ema_bedroom_256_pruned \
--pruning_ratio $1 \
--batch_size 4 \
--pruner random \
--thr 0.05 \
--device cuda:0 \