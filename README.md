# Diff-Pruning: Structural Pruning for Diffusion Models

## Introduction

This work present *Diff-Pruning*, an efficient structrual pruning method for diffusion models. Our empirical assessment, undertaken across four diverse datasets highlights two primary benefits of our proposed method: 1) ``Efficiency``: it enables approximately a 50% reduction in FLOPs at a mere 10% to 20% of the original training expenditure; 2) ``Consistency``: the pruned diffusion models inherently preserve generative behavior congruent with their pre-trained progenitors.

<div align="center">
<img src="assets/LSUN.png" width="90%"></img>
</div>

## Quick Start
### 0. Data & Pretrained Model
Download and extract CIFAR-10 images to *data/cifar10_images*
```bash
python tools/extract_cifar10.py --output data
```

The following script will download an official DDPM model and pack it in the format of Huggingface Diffusers. You can find the converted model at *pretrained/ddpm_ema_cifar10*.
```bash
bash tools/convert_cifar10_ddpm_ema.sh
```

### 1. Pruning
Create a pruned model at *run/pruned/ddpm_cifar10_pruned*
```bash
bash scripts/prune_ddpm_cifar10.sh 0.3  # pruning ratio = 30\%
```

### 2. Finetuning (Post-Training)
Finetune the model and save it as *run/finetuned/ddpm_cifar10_pruned_post_training*
```bash
bash scripts/finetune_ddpm_cifar10.sh
```

### 3. Sampling
**Pruned:** Sampled images from pruned models. All images will be saved to *run/sample/ddpm_cifar10_pruned*
```bash
bash scripts/sample_ddpm_cifar10_pruned.sh
```

**Pretrained:** Sample images from pre-trained models. All images will be saved to *run/sample/ddpm_cifar10_pruned*
```bash
bash scripts/sample_ddpm_cifar10_pretrained.sh
```

Multi-processing sampling are supported. Please refer to [scripts/sample_ddpm_cifar10_pretrained_distributed.sh](scripts/sample_ddpm_cifar10_pretrained_distributed.sh).

### 4. Computing FID score
This script is modified from https://github.com/mseitzer/pytorch-fid. 

```bash
# pre-compute the stats of CIFAR-10 dataset
python fid_score.py --save-stats data/cifar10_images run/fid_stats_cifar10.npz --device cuda:0 --batch-size 256
```

```bash
# Compute the FID score of sampled images
python fid_score.py run/sample/ddpm_cifar10_pruned run/fid_stats_cifar10.npz --device cuda:0 --batch-size 256
```


## Results
<div align="center">
<img src="assets/exp.png" width="70%"></img>
</div>

## Acknowledgement

This project is heavily based on [Diffusers](https://github.com/huggingface/diffusers), [Torch-Pruning](https://github.com/VainF/Torch-Pruning), [pytorch-fid](https://github.com/mseitzer/pytorch-fid). Our experiments were conducted on [ddim](https://github.com/ermongroup/ddim). 

## Citation
If you find this work helpful, please cite:
```
@article{fang2023structural,
  title={Structural pruning for diffusion models},
  author={Fang, Gongfan and Ma, Xinyin and Wang, Xinchao},
  journal={arXiv preprint arXiv:},
  year={2023}
}
```

```
@article{fang2023depgraph,
  title={Depgraph: Towards any structural pruning},
  author={Fang, Gongfan and Ma, Xinyin and Song, Mingli and Mi, Michael Bi and Wang, Xinchao},
  journal={arXiv preprint arXiv:2301.12900},
  year={2023}
}
```
