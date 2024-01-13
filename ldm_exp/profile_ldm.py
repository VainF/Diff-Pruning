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

used = torch.cuda.max_memory_allocated()
print("gpu used {} MB".format(used/1024**2))