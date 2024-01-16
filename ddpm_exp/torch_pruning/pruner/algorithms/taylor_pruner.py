import torch
import math
from .metapruner import MetaPruner
from .scheduler import linear_scheduler
from .. import function
from ..._helpers import _FlattenIndexMapping


class TaylorPruner(MetaPruner):
    def __init__(
        self,
        model,
        example_inputs,
        importance,
        reg=1e-4,
        alpha=4,
        iterative_steps=1,
        iterative_sparsity_scheduler=linear_scheduler,
        ch_sparsity=0.5,
        global_pruning=False,
        channel_groups=dict(),
        max_ch_sparsity=1.0,
        soft_keeping_ratio=0.0,
        ch_sparsity_dict=None,
        round_to=None,
        ignored_layers=None,
        customized_pruners=None,
        unwrapped_parameters=None,
        output_transform=None,
    ):
        super(TaylorPruner, self).__init__(
            model=model,
            example_inputs=example_inputs,
            importance=importance,
            iterative_steps=iterative_steps,
            iterative_sparsity_scheduler=iterative_sparsity_scheduler,
            ch_sparsity=ch_sparsity,
            ch_sparsity_dict=ch_sparsity_dict,
            global_pruning=global_pruning,
            channel_groups=channel_groups,
            max_ch_sparsity=max_ch_sparsity,
            round_to=round_to,
            ignored_layers=ignored_layers,
            customized_pruners=customized_pruners,
            unwrapped_parameters=unwrapped_parameters,
            output_transform=output_transform,
        )
        self.reg = reg
        self.alpha = alpha
        self.groups = list(self.DG.get_all_groups())
        self.soft_keeping_ratio = soft_keeping_ratio
        self.cnt = 0

    @torch.no_grad()
    def regularize(self, model, base=16):
        min_avg = 0
        cnt = 0
        for i, group in enumerate(self.groups):
            ch_groups = self.get_channel_groups(group)
            group_imp = []

            # Get group norm
            for dep, idxs in group:
                idxs.sort()
                layer = dep.target.module
                prune_fn = dep.handler

                # Conv out_channels
                if prune_fn in [
                    function.prune_conv_out_channels,
                    function.prune_linear_out_channels,
                ]:
                    if hasattr(layer, "transposed") and layer.transposed:
                        w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                        dw= layer.weight.grad.data.transpose(1, 0)[idxs].flatten(1)
                    else:
                        w = layer.weight.data[idxs].flatten(1)
                        dw= layer.weight.grad.data[idxs].flatten(1)
                    wdw = w * dw 
                    local_norm = wdw.abs().sum(1)
                    group_imp.append(local_norm)
                
                # Conv in_channels
                elif prune_fn in [
                    function.prune_conv_in_channels,
                    function.prune_linear_in_channels,
                ]:
                    if hasattr(layer, "transposed") and layer.transposed:
                        w = (layer.weight).flatten(1)[idxs]
                        dw= (layer.weight.grad).flatten(1)[idxs]
                    else:
                        w = (layer.weight).transpose(0, 1).flatten(1)[idxs]     
                        dw= (layer.weight.grad).transpose(0, 1).flatten(1)[idxs]     
                    wdw = w * dw
                    local_norm = wdw.abs().sum(1)
                    group_imp.append(local_norm)
                # BN
                elif prune_fn == function.prune_groupnorm_out_channels:
                    # regularize BN
                    if layer.affine:
                        w = layer.weight.data[idxs]
                        dw= layer.weight.grad.data[idxs]
                        wdw = w * dw
                        local_norm = wdw.abs()
                        group_imp.append(local_norm)

            if len(group_imp)==0:
                return None
            imp_size = len(group_imp[0])
            aligned_group_imp = []
            for imp in group_imp:
                if len(imp)==imp_size:
                    aligned_group_imp.append(imp)
            group_imp = torch.stack(aligned_group_imp, dim=0)
            group_imp = group_imp.sum(0).abs()
            min_avg+=float(group_imp.min())
            cnt+=1
            base = 16
            scale = base**((group_imp.max() - group_imp) / (group_imp.max() - group_imp.min()))
            
            # Update Gradient
            for dep, idxs in group:
                layer = dep.target.module
                prune_fn = dep.handler
                if prune_fn in [
                    function.prune_conv_out_channels,
                    function.prune_linear_out_channels,
                ]:
                    w = layer.weight.data[idxs]
                    g = w * scale.view( -1, *([1]*(len(w.shape)-1)) ) #/ group_norm.view( -1, *([1]*(len(w.shape)-1)) ) * group_size #group_size #* scale.view( -1, *([1]*(len(w.shape)-1)) )
                    layer.weight.grad.data[idxs]+=self.reg * g 
                elif prune_fn in [
                    function.prune_conv_in_channels,
                    function.prune_linear_in_channels,
                ]:
                    w = layer.weight.data[:, idxs]
                    g = w * scale.view( 1, -1, *([1]*(len(w.shape)-2))  ) #/ gn.view( 1, -1, *([1]*(len(w.shape)-2)) ) * group_size #* scale.view( 1, -1, *([1]*(len(w.shape)-2))  )
                    layer.weight.grad.data[:, idxs]+=self.reg * g
                elif prune_fn == function.prune_groupnorm_out_channels:
                    # regularize BN
                    if layer.affine is not None:
                        w = layer.weight.data[idxs]
                        g = w * scale #/ group_norm * group_size
                        layer.weight.grad.data[idxs]+=self.reg * g 
        return min_avg / cnt