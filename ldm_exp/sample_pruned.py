import sys
sys.path.append(".")
sys.path.append('./taming-transformers')
from taming.models import vqgan 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--pruned_model", type=str)
parser.add_argument("--finetuned_ckpt", type=str)

args = parser.parse_args()

#@title loading utils
import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config

import torch_pruning as tp

from ldm.models.diffusion.ddim import DDIMSampler

print("Loading model from ", args.pruned_model)
model = torch.load(args.pruned_model, map_location="cpu")
print("Loading finetuned parameters from ", args.finetuned_ckpt)
pl_sd = torch.load(args.finetuned_ckpt, map_location="cpu")
sd = pl_sd["state_dict"]
m, u = model.load_state_dict(sd, strict=False)
model.cuda()
print(model)
sampler = DDIMSampler(model)

n_samples_per_class=1
xc = torch.tensor(n_samples_per_class*[0])
c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
example_inputs = {"x": torch.randn(n_samples_per_class, 3, 64, 64).to(model.device), "timesteps": torch.full((n_samples_per_class,), 1, device=model.device, dtype=torch.long), "context": c}
num_macs, num_params = tp.utils.count_ops_and_params(model.model.diffusion_model, example_inputs)
print("Number of parameters: {}", num_params/1000000, "M")
print("Number of MACs: {}", num_macs/1e9, "G")

import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid


classes = [0, 1, 2, 3] #[25, 187, 448, 992]   # define classes to be sampled here
n_samples_per_class = 6

ddim_steps = 250
ddim_eta = 0.0
scale = 3   # for unconditional guidance

all_samples = list()

with torch.no_grad():
    with model.ema_scope():
        uc = model.get_learned_conditioning(
            {model.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.device)}
            )
        
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
            all_samples.append(x_samples_ddim)


# display as grid
grid = torch.stack(all_samples, 0)
grid = rearrange(grid, 'n b c h w -> (n b) c h w')
grid = make_grid(grid, nrow=n_samples_per_class)

# to image
grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
img = Image.fromarray(grid.astype(np.uint8))
img.save("samples.png")