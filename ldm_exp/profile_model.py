import sys
sys.path.append(".")
sys.path.append('./taming-transformers')
from taming.models import vqgan 
import argparse
from ldm.modules.attention import CrossAttention

parser = argparse.ArgumentParser()
parser.add_argument("--sparsity", type=float, default=0.0)
parser.add_argument("--pruner", type=str, choices=["magnitude", "random", "taylor", "diff-pruning", "reinit", "diff0"], default="magnitude")
args = parser.parse_args()

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

import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid


classes = [25, 187, 448, 992]   # define classes to be sampled here
n_samples_per_class = 6

ddim_steps = 20
ddim_eta = 0.0
scale = 3.0   # for unconditional guidance

print(model)

print("Pruning ...")
model.eval()

if args.pruner == "magnitude":
    imp = tp.importance.MagnitudeImportance()
elif args.pruner == "random":
    imp = tp.importance.RandomImportance()
elif args.pruner == 'taylor':
    imp = tp.importance.TaylorImportance(multivariable=True) # standard first-order taylor expansion
elif args.pruner == 'diff-pruning' or args.pruner == 'diff0':
    imp = tp.importance.TaylorImportance(multivariable=False) # a modified version, estimating the accumulated error of weight removal
else:
    raise ValueError(f"Unknown pruner '{args.pruner}'")

ignored_layers = [model.model.diffusion_model.out]
channel_groups = {}
iterative_steps = 1
uc = model.get_learned_conditioning(
            {model.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.device)}
            )


for m in model.model.diffusion_model.modules():
    if isinstance(m, CrossAttention):
        channel_groups[m.to_q] = m.heads
        channel_groups[m.to_k] = m.heads
        channel_groups[m.to_v] = m.heads


xc = torch.tensor(n_samples_per_class*[classes[0]])
c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
example_inputs = {"x": torch.randn(n_samples_per_class, 3, 64, 64).to(model.device), "timesteps": torch.full((n_samples_per_class,), 1, device=model.device, dtype=torch.long), "context": c}
base_macs, base_params = tp.utils.count_ops_and_params(model.model.diffusion_model, example_inputs)
pruner = tp.pruner.MagnitudePruner(
    model.model.diffusion_model,
    example_inputs,
    importance=imp,
    iterative_steps=1,
    channel_groups =channel_groups,
    ch_sparsity=args.sparsity, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
    ignored_layers=ignored_layers,
    root_module_types=[torch.nn.Conv2d, torch.nn.Linear],
    round_to=2
)
model.zero_grad()

import random
max_loss = -1
for t in range(1000):
    if args.pruner not in ['diff-pruning', 'taylor', 'diff0']:
        break
    xc = torch.tensor(random.sample(range(1000), n_samples_per_class))
    #xc = torch.tensor(n_samples_per_class*[class_label])
    c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                    conditioning=c,
                                    batch_size=n_samples_per_class,
                                    shape=[3, 64, 64],
                                    verbose=False,
                                    unconditional_guidance_scale=scale,
                                    unconditional_conditioning=uc, 
                                    eta=ddim_eta)

    encoded = model.encode_first_stage(samples_ddim)
    example_inputs = {"x": encoded.to(model.device), "timesteps": torch.full((n_samples_per_class,), t, device=model.device, dtype=torch.long), "context": c}
    loss = model.get_loss_at_t(example_inputs['x'], {model.cond_stage_key: xc.to(model.device)}, example_inputs['timesteps'])
    loss = loss[0]
    if loss > max_loss:
        max_loss = loss
    thres = 0.1 if args.pruner == 'diff-pruning' else 0.0
    if args.pruner == 'diff-pruning' or args.pruner == 'diff0':
        if loss / max_loss<thres:
            break
    print(t, (loss / max_loss).item(), loss.item(), max_loss.item())
    loss.backward()
pruner.step() 

print("After pruning")
print(model)

pruend_macs, pruned_params = tp.utils.count_ops_and_params(model.model.diffusion_model, example_inputs)
print(f"MACs: {pruend_macs / base_macs * 100:.2f}%, {base_macs / 1e9:.2f}G => {pruend_macs / 1e9:.2f}G")
print(f"Params: {pruned_params / base_params * 100:.2f}%, {base_params / 1e6:.2f}M => {pruned_params / 1e6:.2f}M")

device = torch.device("cuda")
model.to(device)

#base_macs, base_params = tp.utils.count_ops_and_params(model.model.diffusion_model, example_inputs)
#print("Number of parameters: {}", base_params/1000000, "M")
#print("Number of MACs: {}", base_macs/1e9, "G")

# INIT LOGGERS
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 30
timings=np.zeros((repetitions,1))

#GPU-WARM-UP
for _ in range(10):
    _ = model.model.diffusion_model(**example_inputs)

# MEASURE PERFORMANCE
with torch.no_grad():
    for rep in range(repetitions):
        
        starter.record()
        _ = model.model.diffusion_model(**example_inputs)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        if rep == 0:
            print(torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024, "GB")
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time

mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
print("Inference:")
print("Latency:", mean_syn)
print("FPS:", 1000/mean_syn)
print("-----")


for _ in range(10):
    _ = model.model.diffusion_model(**example_inputs)

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 30
timings=np.zeros((repetitions,1))

for rep in range(repetitions):
    c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
    example_inputs = {"x": torch.randn(n_samples_per_class, 3, 64, 64).to(model.device), "timesteps": torch.full((n_samples_per_class,), 1, device=model.device, dtype=torch.long), "context": c}
    starter.record()
    model.model.diffusion_model.zero_grad()
    output = model.model.diffusion_model(**example_inputs)
    loss = torch.nn.functional.mse_loss(output, torch.randn_like(output))
    loss.backward()
    if rep == 0:
        print(torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024, "GB")
    
    ender.record()
    # WAIT FOR GPU SYNC
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)
    timings[rep] = curr_time

mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
print("Training:")
print("Latency:", mean_syn)
print("FPS:", 1000/mean_syn)
print("-----")

