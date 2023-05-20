from numbers import Number
from typing import Callable
from .metapruner import MetaPruner
from .scheduler import linear_scheduler
import torch
import torch.nn as nn
from .scheduler import linear_scheduler
from ..import function
from ... import ops, dependency

class ScalingFactorPruner(MetaPruner):
    def __init__(
        self,
        model,
        example_inputs,
        importance,
        reg=1e-5,
        iterative_steps=1,
        iterative_sparsity_scheduler: Callable = linear_scheduler,
        ch_sparsity=0.5,
        ch_sparsity_dict=None,
        global_pruning=False,
        max_ch_sparsity=1.0,
        round_to=None,
        channel_groups=None,
        ignored_layers=None,
        customized_pruners=None,
        unwrapped_parameters=None,
        output_transform=None,
    ):
        super(ScalingFactorPruner, self).__init__(
            model=model,
            example_inputs=example_inputs,
            importance=importance,
            iterative_steps=iterative_steps,
            iterative_sparsity_scheduler=iterative_sparsity_scheduler,
            ch_sparsity=ch_sparsity,
            ch_sparsity_dict=ch_sparsity_dict,
            global_pruning=global_pruning,
            max_ch_sparsity=max_ch_sparsity,
            round_to=round_to,
            ignored_layers=ignored_layers,
            customized_pruners=customized_pruners,
            unwrapped_parameters=unwrapped_parameters,
            output_transform=output_transform,
            channel_groups=channel_groups,
        )
        self.reg = reg
        self.groups = list(self.DG.get_all_groups())

    def regularize(self, model):

        for i, group in enumerate(self.groups):
            ch_groups = self.get_channel_groups(group)
            # Get group norm
            #print(group)
            group_norm = []
            for dep, idxs in group:
                idxs.sort()
                layer = dep.target.module
                prune_fn = dep.handler
                if prune_fn == function.prune_groupnorm_out_channels:
                    # regularize BN
                    if layer.affine:
                        w = layer.weight.data[idxs]
                        local_norm = w.pow(2)
                        group_norm.append(local_norm)
            if len(group_norm)==0:
                continue
            group_norm = [gn for gn in group_norm if gn.shape[0]==group_norm[0].shape[0]]
            group_norm = torch.stack(group_norm, 0).sum(0)
            group_norm = group_norm.sqrt()
            base = 16
            scale = base**((group_norm.max() - group_norm) / (group_norm.max() - group_norm.min()))
            
            # Update Gradient
            for dep, idxs in group:
                layer = dep.target.module
                prune_fn = dep.handler
                if prune_fn == function.prune_groupnorm_out_channels and len(idxs)==group_norm.shape[0]:
                    # regularize BN
                    if layer.affine is not None:
                        w = layer.weight.data[idxs]
                        g = w * scale #/ group_norm * group_size
                        layer.weight.grad.data[idxs]+=self.reg * g 

                        #b = layer.bias.data[idxs]
                        #g = b * scale #/ group_norm * group_size
                        #layer.bias.grad.data[idxs]+=self.reg * g 
