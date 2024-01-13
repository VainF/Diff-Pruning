import sys, os
sys.path.append(".")
sys.path.append('./taming-transformers')
from taming.models import vqgan 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--pruned_model", type=str, default=None)
parser.add_argument("--finetuned_ckpt", type=str, default=None)
parser.add_argument("--ipc", type=int, default=50)
parser.add_argument("--output", type=str, default='run')
parser.add_argument("--batch_size", type=int, default=50)

args = parser.parse_args()

#@title loading utils
import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config

import torch_pruning as tp

from ldm.models.diffusion.ddim import DDIMSampler

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

def get_model():
    config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")  
    model = load_model_from_config(config, "models/ldm/cin256-v2/model.ckpt")
    return model

if args.pruned_model is None:
    model = get_model()
else:
    print("Loading model from ", args.pruned_model)
    model = torch.load(args.pruned_model, map_location="cpu")
    print("Loading finetuned parameters from ", args.finetuned_ckpt)
    pl_sd = torch.load(args.finetuned_ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    m, u = model.load_state_dict(sd, strict=False)
model.cuda()
print(model)
sampler = DDIMSampler(model)

num_params = sum(p.numel() for p in model.parameters())
print("Number of parameters: {}", num_params/1000000, "M")

import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid


classes = range(1000)   # define classes to be sampled here
n_samples_per_class = args.batch_size
n_batch_per_class = args.ipc // args.batch_size

ddim_steps = 250
ddim_eta = 0.0
scale = 3.0   # for unconditional guidance

all_samples = list()

from torchvision import utils as tvu
os.makedirs(args.output, exist_ok=True)

img_id = 0
with torch.no_grad():
    with model.ema_scope():
        uc = model.get_learned_conditioning(
            {model.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.device)}
        )
        
        for _ in range(n_batch_per_class):
            for class_label in classes:
                print(f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
                xc = torch.tensor(n_samples_per_class*[class_label])
                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                conditioning=c,
                                                batch_size=n_samples_per_class,
                                                shape=[3, 64, 64],
                                                verbose=False,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc, 
                                                eta=ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                            min=0.0, max=1.0)
                #all_samples.append(x_samples_ddim)
                for i in range(len(x_samples_ddim)):
                    tvu.save_image(
                        x_samples_ddim[i], os.path.join(args.output, f"{class_label}_{img_id}.png")
                    )
                    img_id += 1
                