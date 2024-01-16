import abc
import torch
import torch.nn as nn

import typing
from .pruner import function
from ._helpers import _FlattenIndexMapping
from . import ops
import math

class Importance(abc.ABC):
    """ estimate the importance of a Pruning Group, and return an 1-D per-channel importance score.
    """
    @abc.abstractclassmethod
    def __call__(self, group)-> torch.Tensor:
        raise NotImplementedError

class MagnitudeImportance(Importance):
    def __init__(self, p=2, group_reduction="mean", normalizer='mean'):
        self.p = p
        self.group_reduction = group_reduction
        self.normalizer = normalizer

    def _normalize(self, group_importance, normalizer):
        if normalizer is None: 
            return group_importance
        elif isinstance(normalizer, typing.Callable):
            return normalizer(group_importance)
        elif normalizer == "sum":
            return group_importance / group_importance.sum()
        elif normalizer == "standarization":
            return (group_importance - group_importance.min()) / (group_importance.max() - group_importance.min()+1e-8)
        elif normalizer == "mean":
            return group_importance / group_importance.mean()
        elif normalizer == "max":
            return group_importance / group_importance.max()
        elif normalizer == 'gaussian':
            return (group_importance - group_importance.mean()) / (group_importance.std()+1e-8)
        else:
            raise NotImplementedError
        
    def _reduce(self, group_imp):
        if self.group_reduction == "sum":
            group_imp = group_imp.sum(dim=0)
        elif self.group_reduction == "mean":
            group_imp = group_imp.mean(dim=0)
        elif self.group_reduction == "max":
            group_imp = group_imp.max(dim=0)[0]
        elif self.group_reduction == "prod":
            group_imp = torch.prod(group_imp, dim=0)
        elif self.group_reduction=='first':
            group_imp = group_imp[0]
        elif self.group_reduction is None:
            group_imp = group_imp
        else: 
            raise NotImplementedError
        return group_imp

    @torch.no_grad()
    def __call__(self, group, ch_groups=1):
        group_imp = []
        #Get group norm
        #print(group.details())
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
                else:
                    w = layer.weight.data[idxs].flatten(1)
                local_norm = w.abs().pow(self.p).sum(1)
                #if ch_groups>1:
                #    local_norm = local_norm.view(ch_groups, -1).sum(0)
                #    local_norm = local_norm.repeat(ch_groups)
                group_imp.append(local_norm)

            # Conv in_channels
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                is_conv_flatten_linear = False
                if hasattr(layer, "transposed") and layer.transposed:
                    w = (layer.weight).flatten(1)
                else:
                    w = (layer.weight).transpose(0, 1).flatten(1)     
                
                #if ch_groups>1 and prune_fn==function.prune_conv_in_channels and layer.groups==1:
                    # standard convs with ch_groups>1
                    #print(w.shape)
                #    w = w.view(ch_groups, w.shape[0] // ch_groups, w.shape[1]).flatten(1)       
                local_norm = w.abs().pow(self.p).sum(1)
                #if ch_groups>1:
                    #if len(local_norm)==len(group_imp[0]):
                    #local_norm = local_norm.view(ch_groups, -1).sum(0)
                #    local_norm = local_norm.repeat(ch_groups)
                local_norm = local_norm[idxs]
                group_imp.append(local_norm)
            # BN
            elif prune_fn == function.prune_batchnorm_out_channels:
                # regularize BN
                if layer.affine:
                    w = layer.weight.data[idxs]
                    local_norm = w.abs().pow(self.p)
                    #if ch_groups>1:
                    #    local_norm = local_norm.view(ch_groups, -1).sum(0)
                    #    local_norm = local_norm.repeat(ch_groups)
                    #print(local_norm.shape)
                    group_imp.append(local_norm)
        if len(group_imp)==0:
            return None
        imp_size = len(group_imp[0])
        aligned_group_imp = []
        for imp in group_imp:
            if len(imp)==imp_size:
                aligned_group_imp.append(imp)
        group_imp = torch.stack(aligned_group_imp, dim=0)
        group_imp = self._reduce(group_imp)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp


class BNScaleImportance(MagnitudeImportance):
    """Learning Efficient Convolutional Networks through Network Slimming, 
    https://arxiv.org/abs/1708.06519
    """
    def __init__(self, group_reduction='mean', normalizer='mean'):
        super().__init__(p=1, group_reduction=group_reduction, normalizer=normalizer)
    
    def __call__(self, group, ch_groups=1):
        group_imp = []
        for dep, _ in group:
            module = dep.target.module
            if isinstance(module, (ops.TORCH_BATCHNORM)) and module.affine:
                local_imp = torch.abs(module.weight.data)
                if ch_groups>1:
                    local_imp = local_imp.view(ch_groups, -1).mean(0)
                    local_imp = local_imp.repeat(ch_groups)
                group_imp.append(local_imp)
        if len(group_imp)==0:
            return None
        group_imp = torch.stack(group_imp, dim=0)
        group_imp = self._reduce(group_imp)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp


class LAMPImportance(MagnitudeImportance):
    """Layer-adaptive Sparsity for the Magnitude-based Pruning,
    https://arxiv.org/abs/2010.07611
    """
    def __init__(self, p=2, group_reduction="mean", normalizer='mean'):
        super().__init__(p=p, group_reduction=group_reduction, normalizer=normalizer)

    @torch.no_grad()
    def __call__(self, group, **kwargs):
        group_imp = []
        for dep, idxs in group:
            layer = dep.target.module
            prune_fn = dep.handler

            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = (layer.weight)[:, idxs].transpose(0, 1)
                else:
                    w = (layer.weight)[idxs]
                local_imp = torch.norm(
                    torch.flatten(w, 1), dim=1, p=self.p)
                group_imp.append(local_imp)

            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = (layer.weight)[idxs].flatten(1)
                else:
                    w = (layer.weight)[:, idxs].transpose(0, 1).flatten(1)
                if (
                    w.shape[0] != group_imp[0].shape[0]
                ):  # for conv-flatten-linear without global pooling
                    w = w.view(
                        group_imp[0].shape[0],
                        w.shape[0] // group_imp[0].shape[0],
                        w.shape[1],
                    ).flatten(1)
                local_imp = torch.norm(w, dim=1, p=self.p)
                group_imp.append(local_imp)

            elif prune_fn == function.prune_batchnorm_out_channels:
                if layer.affine is not None:
                    w = (layer.weight)[idxs].view(-1, 1)
                    local_imp = torch.norm(w, dim=1, p=self.p)
                    group_imp.append(local_imp)
        if len(group_imp)==0:
            return None
        group_imp = torch.stack(group_imp, dim=0)
        group_imp = self._reduce(group_imp)
        group_imp = self._normalize(group_imp, self.normalizer)
        return self.lamp(group_imp)

    def lamp(self, imp):
        argsort_idx = torch.argsort(imp, dim=0, descending=True).tolist()
        sorted_imp = imp[argsort_idx]
        cumsum_imp = torch.cumsum(sorted_imp, dim=0)
        sorted_imp = sorted_imp / cumsum_imp
        inversed_idx = torch.arange(len(sorted_imp))[
            argsort_idx
        ].tolist()  # [0, 1, 2, 3, ..., ]
        return sorted_imp[inversed_idx]

class RandomImportance(Importance):
    @torch.no_grad()
    def __call__(self, group, **kwargs):
        _, idxs = group[0]
        return torch.rand(len(idxs))

class GroupNormImportance(MagnitudeImportance):
    def __init__(self, p=2, normalizer='max'):
        super().__init__(p=p, group_reduction=None, normalizer=normalizer)
        self.p = p
        self.normalizer = normalizer
        
    @torch.no_grad()
    def __call__(self, group, ch_groups=1):
        group_norm = 0

        #Get group norm
        for dep, idxs in group:
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler

            # Conv out_channels
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:  
                if hasattr(layer, 'transposed') and layer.transposed:
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                else:
                    w = layer.weight.data[idxs].flatten(1)
                local_norm = w.abs().pow(self.p).sum(1)
                #print(local_norm.shape, layer, idxs, ch_groups)
                if ch_groups>1:
                    local_norm = local_norm.view(ch_groups, -1).sum(0)
                    local_norm = local_norm.repeat(ch_groups)
                group_norm+=local_norm
                #if layer.bias is not None:
                #    group_norm += layer.bias.data[idxs].pow(2)
            # Conv in_channels
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                is_conv_flatten_linear = False
                if hasattr(layer, 'transposed') and layer.transposed:
                    w = (layer.weight).flatten(1)  
                else:
                    w = (layer.weight).transpose(0, 1).flatten(1)             
                if (w.shape[0] != group_norm.shape[0]):  
                    if (hasattr(dep, 'index_mapping') and isinstance(dep.index_mapping, _FlattenIndexMapping)):
                        #conv-flatten
                        w = w[idxs].view(
                            group_norm.shape[0],
                            w.shape[0] // group_norm.shape[0],
                            w.shape[1],
                        ).flatten(1)
                        is_conv_flatten_linear = True
                    elif ch_groups>1 and prune_fn==function.prune_conv_in_channels and layer.groups==1:
                        # non-grouped conv with group convs
                        w = w.view(w.shape[0] // group_norm.shape[0],
                                group_norm.shape[0], w.shape[1]).transpose(0, 1).flatten(1)           
                local_norm = w.abs().pow(self.p).sum(1)
                if ch_groups>1:
                    if len(local_norm)==len(group_norm):
                        local_norm = local_norm.view(ch_groups, -1).sum(0)
                    local_norm = local_norm.repeat(ch_groups)
                if not is_conv_flatten_linear:
                    local_norm = local_norm[idxs]
                group_norm += local_norm
            # BN
            elif prune_fn == function.prune_batchnorm_out_channels:
                # regularize BN
                if layer.affine:
                    w = layer.weight.data[idxs]
                    local_norm = w.abs().pow(self.p)
                    if ch_groups>1:
                        local_norm = local_norm.view(ch_groups, -1).sum(0)
                        local_norm = local_norm.repeat(ch_groups)
                    group_norm += local_norm

            elif prune_fn == function.prune_lstm_out_channels:
                _idxs = torch.tensor(idxs)
                local_norm = 0
                local_norm_reverse = 0
                num_layers = layer.num_layers
                expanded_idxs = torch.cat([ _idxs+i*layer.hidden_size for i in range(4) ], dim=0)
                if layer.bidirectional:
                    postfix = ['', '_reverse']
                else:
                    postfix = ['']

                local_norm+=getattr(layer, 'weight_hh_l0')[expanded_idxs].abs().pow(self.p).sum(1).view(4, -1).sum(0)
                local_norm+=getattr(layer, 'weight_hh_l0')[:, _idxs].abs().pow(self.p).sum(0)
                local_norm+=getattr(layer, 'weight_ih_l0')[expanded_idxs].abs().pow(self.p).sum(1).view(4, -1).sum(0)
                if layer.bidirectional:
                    local_norm_reverse+=getattr(layer, 'weight_hh_l0')[expanded_idxs].abs().pow(self.p).sum(1).view(4, -1).sum(0)
                    local_norm_reverse+=getattr(layer, 'weight_hh_l0')[:, _idxs].abs().pow(self.p).sum(0)
                    local_norm_reverse+=getattr(layer, 'weight_ih_l0')[expanded_idxs].abs().pow(self.p).sum(1).view(4, -1).sum(0)
                    local_norm = torch.cat([local_norm, local_norm_reverse], dim=0)
                group_norm += local_norm
            elif prune_fn == function.prune_lstm_in_channels:
                local_norm=getattr(layer, 'weight_ih_l0')[:, idxs].abs().pow(self.p).sum(0)
                if layer.bidirectional:
                    local_norm_reverse+=getattr(layer, 'weight_ih_l0_reverse')[:, idxs].abs().pow(self.p).sum(0)
                    local_norm = torch.cat([local_norm, local_norm_reverse], dim=0)
                group_norm+=local_norm
        group_imp = group_norm**(1/self.p)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp 

class TaylorImportance(Importance):
    def __init__(self, group_reduction="mean", normalizer='mean'):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
    
    def set_model(self, model):
        self.model = model

    def _normalize(self, group_importance, normalizer):
        if normalizer is None: 
            return group_importance
        elif isinstance(normalizer, typing.Callable):
            return normalizer(group_importance)
        elif normalizer == "sum":
            return group_importance / group_importance.sum()
        elif normalizer == "standarization":
            return (group_importance - group_importance.min()) / (group_importance.max() - group_importance.min()+1e-8)
        elif normalizer == "mean":
            return group_importance / group_importance.mean()
        elif normalizer == "max":
            return group_importance / group_importance.max()
        elif normalizer == 'gaussian':
            return (group_importance - group_importance.mean()) / (group_importance.std()+1e-8)
        else:
            raise NotImplementedError
        
    def _reduce(self, group_imp):
        if self.group_reduction == "sum":
            group_imp = group_imp.sum(dim=0)
        elif self.group_reduction == "mean":
            group_imp = group_imp.mean(dim=0)
        elif self.group_reduction == "max":
            group_imp = group_imp.max(dim=0)[0]
        elif self.group_reduction == "prod":
            group_imp = torch.prod(group_imp, dim=0)
        elif self.group_reduction=='first':
            group_imp = group_imp[0]
        elif self.group_reduction is None:
            group_imp = group_imp
        else: 
            raise NotImplementedError
        return group_imp

    @torch.no_grad()
    def __call__(self, group, ch_groups=1):
        group_imp = []
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
                wdw = (w*dw).abs().pow(2).sum(1)
                local_norm = wdw
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
                wdw = (w*dw).abs().pow(2).sum(1)
                local_norm = wdw
                group_imp.append(local_norm)
            # BN
            elif prune_fn == function.prune_groupnorm_out_channels:
                # regularize BN
                if layer.affine:
                    w = layer.weight.data[idxs]
                    dw= layer.weight.grad.data[idxs]
                    wdw = (w*dw).abs()
                    local_norm = wdw
                    group_imp.append(local_norm)

        if len(group_imp)==0:
            return None
        imp_size = len(group_imp[0])
        aligned_group_imp = []
        for imp in group_imp:
            if len(imp)==imp_size:
                aligned_group_imp.append(imp)
        group_imp = torch.stack(aligned_group_imp, dim=0)
        group_imp = group_imp.sum(0)  #self._reduce(group_imp)
        #if ch_groups>1:
        #    group_imp = group_imp.view(ch_groups, -1).mean(0)
        #    group_imp = group_imp.repeat(ch_groups)
        #group_imp = group_imp.abs()
        #group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp



class FullTaylorImportance(Importance):
    def __init__(self, order=1, group_reduction="mean", normalizer='mean'):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.order=order

    def set_model(self, model):
        self.model = model

    def _normalize(self, group_importance, normalizer):
        if normalizer is None: 
            return group_importance
        elif isinstance(normalizer, typing.Callable):
            return normalizer(group_importance)
        elif normalizer == "sum":
            return group_importance / group_importance.sum()
        elif normalizer == "standarization":
            return (group_importance - group_importance.min()) / (group_importance.max() - group_importance.min()+1e-8)
        elif normalizer == "mean":
            return group_importance / group_importance.mean()
        elif normalizer == "max":
            return group_importance / group_importance.max()
        elif normalizer == 'gaussian':
            return (group_importance - group_importance.mean()) / (group_importance.std()+1e-8)
        else:
            raise NotImplementedError
        
    def _reduce(self, group_imp):
        if self.group_reduction == "sum":
            group_imp = group_imp.sum(dim=0)
        elif self.group_reduction == "mean":
            group_imp = group_imp.mean(dim=0)
        elif self.group_reduction == "max":
            group_imp = group_imp.max(dim=0)[0]
        elif self.group_reduction == "prod":
            group_imp = torch.prod(group_imp, dim=0)
        elif self.group_reduction=='first':
            group_imp = group_imp[0]
        elif self.group_reduction is None:
            group_imp = group_imp
        else: 
            raise NotImplementedError
        return group_imp

    @torch.no_grad()
    def __call__(self, group, ch_groups=1):
        group_imp = []
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
    
                if self.order == 1:
                    wdw = (w*dw).sum(1)
                elif self.order == 2:
                    wdw = (w*dw).sum(1) + (w*dw).pow(2).sum(1)
                
                local_norm = wdw
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
                if self.order == 1:
                    wdw = (w*dw).sum(1)
                elif self.order == 2:
                    wdw = (w*dw).sum(1) + (w*dw).pow(2).sum(1)
                
                local_norm = wdw
                group_imp.append(local_norm)
            # BN
            elif prune_fn == function.prune_groupnorm_out_channels:
                # regularize BN
                if layer.affine:
                    w = layer.weight.data[idxs]
                    dw= layer.weight.grad.data[idxs]
                    if self.order == 1:
                        wdw = (w*dw)
                    elif self.order == 2:
                        wdw = (w*dw) + (w*dw).pow(2)
                    local_norm = wdw
                    group_imp.append(local_norm)

        if len(group_imp)==0:
            return None
        imp_size = len(group_imp[0])
        aligned_group_imp = []
        for imp in group_imp:
            if len(imp)==imp_size:
                aligned_group_imp.append(imp)
        group_imp = torch.stack(aligned_group_imp, dim=0)
        group_imp = group_imp.sum(0).abs()  #self._reduce(group_imp)
        return group_imp




class AbsTaylorImportance(Importance):
    def __init__(self, order=1, group_reduction="mean", normalizer='mean'):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.order=order
        self._accum_abs_grad = {}
        
    def set_model(self, model):
        self.model = model

    def _normalize(self, group_importance, normalizer):
        if normalizer is None: 
            return group_importance
        elif isinstance(normalizer, typing.Callable):
            return normalizer(group_importance)
        elif normalizer == "sum":
            return group_importance / group_importance.sum()
        elif normalizer == "standarization":
            return (group_importance - group_importance.min()) / (group_importance.max() - group_importance.min()+1e-8)
        elif normalizer == "mean":
            return group_importance / group_importance.mean()
        elif normalizer == "max":
            return group_importance / group_importance.max()
        elif normalizer == 'gaussian':
            return (group_importance - group_importance.mean()) / (group_importance.std()+1e-8)
        else:
            raise NotImplementedError
        
    def _reduce(self, group_imp):
        if self.group_reduction == "sum":
            group_imp = group_imp.sum(dim=0)
        elif self.group_reduction == "mean":
            group_imp = group_imp.mean(dim=0)
        elif self.group_reduction == "max":
            group_imp = group_imp.max(dim=0)[0]
        elif self.group_reduction == "prod":
            group_imp = torch.prod(group_imp, dim=0)
        elif self.group_reduction=='first':
            group_imp = group_imp[0]
        elif self.group_reduction is None:
            group_imp = group_imp
        else: 
            raise NotImplementedError
        return group_imp

    def accum_abs_grad(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                if p not in self._accum_abs_grad:
                    self._accum_abs_grad[name]=(p.grad).abs()
                else:
                    self._accum_abs_grad[name]+=(p.grad).abs()    
    
    def assign_abs_grad(self, model):
        for name, p in model.named_parameters():
            if name in self._accum_abs_grad:
                p.grad = self._accum_abs_grad[name]

    @torch.no_grad()
    def __call__(self, group, ch_groups=1):
        group_imp = []
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
                wdw = (w*dw).abs().sum(1)
                local_norm = wdw
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
                wdw = (w * dw).abs().sum(1)
                local_norm = wdw
                group_imp.append(local_norm)
            # BN
            elif prune_fn == function.prune_groupnorm_out_channels:
                # regularize BN
                if layer.affine:
                    w = layer.weight.data[idxs]
                    dw= layer.weight.grad.data[idxs]
                    wdw = (w * dw).abs()
                    local_norm = wdw
                    group_imp.append(local_norm)

        if len(group_imp)==0:
            return None
        imp_size = len(group_imp[0])
        aligned_group_imp = []
        for imp in group_imp:
            if len(imp)==imp_size:
                aligned_group_imp.append(imp)
        group_imp = torch.stack(aligned_group_imp, dim=0)
        group_imp = group_imp.sum(0)  #self._reduce(group_imp)
        #if ch_groups>1:
        #    group_imp = group_imp.view(ch_groups, -1).mean(0)
        #    group_imp = group_imp.repeat(ch_groups)
        #group_imp = group_imp.abs()
        #group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp

class FisherImportance(Importance):
    def __init__(self, group_reduction="mean", normalizer='mean'):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
    
    def set_model(self, model):
        self.model = model

    def _normalize(self, group_importance, normalizer):
        if normalizer is None: 
            return group_importance
        elif isinstance(normalizer, typing.Callable):
            return normalizer(group_importance)
        elif normalizer == "sum":
            return group_importance / group_importance.sum()
        elif normalizer == "standarization":
            return (group_importance - group_importance.min()) / (group_importance.max() - group_importance.min()+1e-8)
        elif normalizer == "mean":
            return group_importance / group_importance.mean()
        elif normalizer == "max":
            return group_importance / group_importance.max()
        elif normalizer == 'gaussian':
            return (group_importance - group_importance.mean()) / (group_importance.std()+1e-8)
        else:
            raise NotImplementedError
        
    def _reduce(self, group_imp):
        if self.group_reduction == "sum":
            group_imp = group_imp.sum(dim=0)
        elif self.group_reduction == "mean":
            group_imp = group_imp.mean(dim=0)
        elif self.group_reduction == "max":
            group_imp = group_imp.max(dim=0)[0]
        elif self.group_reduction == "prod":
            group_imp = torch.prod(group_imp, dim=0)
        elif self.group_reduction=='first':
            group_imp = group_imp[0]
        elif self.group_reduction is None:
            group_imp = group_imp
        else: 
            raise NotImplementedError
        return group_imp

    @torch.no_grad()
    def __call__(self, group, ch_groups=1):
        group_imp = []
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
                #wdw = dw
                local_norm = dw.pow(2).sum(1)
                #if ch_groups>1:
                #    local_norm = local_norm.view(ch_groups, -1).mean(0)
                #    local_norm = local_norm.repeat(ch_groups)
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

                local_norm = dw.pow(2).sum(1)
                #if ch_groups>1:
                #    local_norm = local_norm.view(ch_groups, -1).mean(0) 
                #    local_norm = local_norm.repeat(ch_groups)    
                group_imp.append(local_norm)
            # BN
            elif prune_fn == function.prune_groupnorm_out_channels:
                # regularize BN
                if layer.affine:
                    w = layer.weight.data[idxs]
                    dw= layer.weight.grad.data[idxs]
                    wdw = w*dw
                    local_norm = wdw.pow(2)
                    #if ch_groups>1:
                    #    local_norm = local_norm.view(ch_groups, -1).mean(0)
                    #    local_norm = local_norm.repeat(ch_groups)
                    group_imp.append(local_norm)
        if len(group_imp)==0:
            return None
        imp_size = len(group_imp[0])
        aligned_group_imp = []
        for imp in group_imp:
            if len(imp)==imp_size:
                aligned_group_imp.append(imp)
        group_imp = torch.stack(aligned_group_imp, dim=0)
        group_imp = group_imp.sum(0)  #self._reduce(group_imp)
        #if ch_groups>1:
        #    group_imp = group_imp.view(ch_groups, -1).mean(0)
        #    group_imp = group_imp.repeat(ch_groups)
        #group_imp = group_imp.abs()
        #group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp