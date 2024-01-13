import sys
sys.path.append(".")
sys.path.append('./taming-transformers')
from taming.models import vqgan 
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

#@title loading utils
import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config

import torch_pruning as tp

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

from ldm.models.diffusion.ddim import DDIMSampler

model = get_model()
sampler = DDIMSampler(model)

n_samples_per_class=1
xc = torch.tensor(n_samples_per_class*[0])
c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
example_inputs = {"x": torch.randn(n_samples_per_class, 3, 64, 64).to(model.device), "timesteps": torch.full((n_samples_per_class,), 1, device=model.device, dtype=torch.long), "context": c}
num_macs, num_params = tp.utils.count_ops_and_params(model.model.diffusion_model, example_inputs)


with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True, record_shapes=True) as prof:
    model.model.diffusion_model(**example_inputs)

print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))