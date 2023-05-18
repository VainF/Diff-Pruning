python ddpm_sample.py \
--output_dir run/cifar10_sample \
--model_path pretrained/ddpm_ema_cifar10 \
--batch_size 128 \
--pruned_model_ckpt run/pruned/ddpm_cifar10_pruned/pruned/unet_pruned.pth \