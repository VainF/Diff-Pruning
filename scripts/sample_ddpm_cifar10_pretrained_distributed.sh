python -m torch.distributed.launch --nproc_per_node=8 --master_port 22222 --use_env ddpm_sample.py \
--output_dir run/sample/ddpm_cifar10_pretrained \
--model_path pretrained/ddpm_ema_cifar10 \
--batch_size 128 \