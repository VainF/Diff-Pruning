python ddpm_sample.py \
--output_dir run/sample/cifar10_pruned \
--model_path pretrained/ddpm_ema_cifar10 \
--batch_size 128 \
--pruned_model_ckpt run/finetuned/ddpm_cifar10_pruned_post_training/pruned/unet_ema_pruned.pth \