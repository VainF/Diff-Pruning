import torch.nn as nn


class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = []

    def to(self, device=None) -> None:
        self.shadow = [
            p.to(device=device)
            for p in self.shadow
        ]

    def copy_to(self, parameters) -> None:
        parameters = list(parameters)
        for s_param, param in zip(self.shadow, parameters):
            param.data.copy_(s_param.to(param.device).data)

    def store(self, parameters) -> None:
        r"""
        Args:
        Save the current parameters for restoring later.
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                temporarily stored.
        """
        self.temp_stored_params = [param.detach().cpu().clone() for param in parameters]

    def restore(self, parameters) -> None:
        if self.temp_stored_params is None:
            raise RuntimeError("This ExponentialMovingAverage has no `store()`ed weights " "to `restore()`")
        for c_param, param in zip(self.temp_stored_params, parameters):
            param.data.copy_(c_param.data)
        self.temp_stored_params = None

    def register(self, module):
        for param in module.parameters():
            if param.requires_grad:
                self.shadow.append(param.data.clone())

    def update(self, module):
        for i, (shadow_param, param) in enumerate(zip(self.shadow, module.parameters())):
            if param.requires_grad:
                shadow_param.data = (
                    1. - self.mu) * param.data + self.mu * shadow_param.data
                #if i==0:
                #    print(shadow_param.flatten()[0])
                    
    def ema(self, module):
        for shadow_param, param in zip(self.shadow, module.parameters()):
            if param.requires_grad:
                param.data.copy_(shadow_param.data)

    #def ema_copy(self, module):
    #    if isinstance(module, nn.parallel.DistributedDataParallel):
    #        from copy import deepcopy
    #        inner_module = module.module
    #        module_copy = deepcopy(inner_module).to(inner_module.config.device)
    ##        module_copy.load_state_dict(inner_module.state_dict())
    #        module_copy = nn.DistributedDataParallel(module_copy)
    #    else:
    #        module_copy = deepcopy(inner_module).to(module.config.device)
    #        module_copy.load_state_dict(module.state_dict())
    #    # module_copy = copy.deepcopy(module)
    #    self.ema(module_copy)
    #    return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        if isinstance(state_dict, list):
            self.shadow = state_dict
        else:
            self.shadow = state_dict.values()