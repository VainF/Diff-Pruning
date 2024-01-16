import torch

state = torch.load("model.ckpt.old")
old_dict = state[0]
print(state[0].keys())
state[0] = {pname.replace("module.", ''): pval for pname, pval in old_dict.items()}
print(state[0].keys())
torch.save(state, "model.ckpt")