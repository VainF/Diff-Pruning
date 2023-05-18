mkdir -p pretrained
wget https://heibox.uni-heidelberg.de/f/2e4f01e2d9ee49bab1d5/?dl=1 -O pretrained/cifar10-ema-model-790000.ckpt
python tools/convert_ddpm_original_checkpoint_to_diffusers_cifar10.py --checkpoint_path pretrained/cifar10-ema-model-790000.ckpt --config_file tools/ddpm_cifar10_config.json --dump_path pretrained/ddpm_ema_cifar10
