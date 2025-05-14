# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""This code is copied fron NVIDIA apex:
      https://github.com/NVIDIA/apex
   with some changes. """

import numbers
import os
import torch
from torch.nn.parameter import Parameter
from torch.nn import init
import importlib

from megatron.core.utils import make_viewless_tensor

try:
    from apex.contrib.layer_norm.layer_norm import FastLayerNormFN
    from apex.contrib.layer_norm.layer_norm import FastRMSNormFN
    HAVE_PERSIST_LAYER_NORM = True
except:
    HAVE_PERSIST_LAYER_NORM = False

try:
    from transformer_engine.pytorch.module.rmsnorm import _RMSNorm as TERMSNormFN
    HAVE_TE_RMS_NORM = True
except ModuleNotFoundError:
    raise


from apex.normalization.fused_layer_norm import FusedLayerNormAffineFunction, FusedRMSNormAffineFunction


global fused_layer_norm_cuda
fused_layer_norm_cuda = None


class MixedFusedLayerNorm(torch.nn.Module):

  def __init__(self, normalized_shape, eps=1e-5,
               no_persist_layer_norm=True,
               sequence_parallel=False,
               apply_layernorm_1p=False):
        super(MixedFusedLayerNorm, self).__init__()

        self.apply_layernorm_1p = apply_layernorm_1p

        global fused_layer_norm_cuda
        fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")

        # List of hiddens sizes supported in the persistent layer norm kernel
        # If the hidden size is not supported, fall back to the non-persistent
        # kernel.
        persist_ln_hidden_sizes = [1024, 1536, 2048, 2304, 3072, 3840, 4096,
            5120, 6144, 8192, 10240, 12288, 12800, 15360, 16384, 18432, 20480,
            24576, 25600, 30720, 32768, 40960, 49152, 65536]
        if normalized_shape not in persist_ln_hidden_sizes or \
                not HAVE_PERSIST_LAYER_NORM:
            no_persist_layer_norm = True

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = Parameter(torch.Tensor(*normalized_shape))
        self.bias = Parameter(torch.Tensor(*normalized_shape))
        self.reset_parameters()
        self.no_persist_layer_norm = no_persist_layer_norm
        self.sequence_parallel = sequence_parallel

        # set sequence parallelism flag on weight and bias parameters
        setattr(self.weight, 'sequence_parallel', self.sequence_parallel)
        setattr(self.bias, 'sequence_parallel', self.sequence_parallel)


  def reset_parameters(self):

    if self.apply_layernorm_1p:
        init.zeros_(self.weight)
        init.zeros_(self.bias)
    else:
        init.ones_(self.weight)
        init.zeros_(self.bias)

  def forward(self, input):

    weight = self.weight + 1 if self.apply_layernorm_1p else self.weight

    if self.no_persist_layer_norm:
        return FusedLayerNormAffineFunction.apply(input, weight, self.bias, self.normalized_shape, self.eps)
    else:
        output = FastLayerNormFN.apply(input, weight, self.bias, self.eps)

        # Apex's fast layer norm function outputs a 'view' tensor (i.e., has
        # a populated '_base' field). This will result in schedule.py's
        # deallocate_output_tensor() throwing an error, so a viewless tensor is
        # created to prevent this.
        output = make_viewless_tensor(inp = output,
                                      requires_grad = input.requires_grad,
                                      keep_graph = True)

        return output

class MixedFusedRMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5,
               no_persist_rms_norm = True,
               sequence_parallel=False,
               apply_layernorm_1p=False):
        super(MixedFusedRMSNorm, self).__init__()

        self.apply_layernorm_1p = apply_layernorm_1p

        global fused_layer_norm_cuda
        if fused_layer_norm_cuda is None:
            fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")

        persist_rn_hidden_sizes = [768, 1024, 1536, 2048, 2304, 3072, 3840, 
            4096, 5120, 6144, 8192, 10240, 12288, 12800, 14336, 15360, 16384, 
            18432, 20480, 24576, 25600, 30720, 32768, 40960, 49152, 65536,
        ]
        if normalized_shape not in persist_rn_hidden_sizes or \
                not HAVE_PERSIST_LAYER_NORM:
            no_persist_rms_norm = True

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.no_persist_rms_norm = no_persist_rms_norm
        self.eps = eps
        self.weight = Parameter(torch.Tensor(*normalized_shape))
        self.reset_parameters()
        self.sequence_parallel = sequence_parallel

        self.fwd_rmsnorm_sm_margin = int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0"))
        self.bwd_rmsnorm_sm_margin = int(os.getenv("NVTE_BWD_LAYERNORM_SM_MARGIN", "0"))

        # set sequence parallelism flag on weight and bias parameters
        setattr(self.weight, 'sequence_parallel', self.sequence_parallel)


    def reset_parameters(self):

        if self.apply_layernorm_1p:
            init.zeros_(self.weight)
        else:
            init.ones_(self.weight)

    def forward(self, input):

        weight = self.weight + 1 if self.apply_layernorm_1p else self.weight

        if HAVE_TE_RMS_NORM:
            return TERMSNormFN.apply(input, weight, self.eps, self.fwd_rmsnorm_sm_margin, self.bwd_rmsnorm_sm_margin,
                                     self.apply_layernorm_1p, torch.is_grad_enabled(), input.dtype)

        if self.no_persist_rms_norm:
            return FusedRMSNormAffineFunction.apply(input, weight, self.normalized_shape, self.eps)
        else:
            output = FastRMSNormFN.apply(input, weight, self.eps)
            # Apex's fast rms norm function outputs a 'view' tensor (i.e., has
            # a populated '_base' field). This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            output = make_viewless_tensor(inp = output,
                                        requires_grad = input.requires_grad,
                                        keep_graph = True)
            return output
