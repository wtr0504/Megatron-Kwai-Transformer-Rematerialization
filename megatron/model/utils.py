# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Utilities for models."""

import math

import torch

from megatron import get_args
from megatron.core import mpu
from megatron.core.context_parallel import dattention

def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method_normal(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


def get_linear_layer(rows, columns, init_method):
    """Simple linear layer with weight initialization."""
    layer = torch.nn.Linear(rows, columns)
    if get_args().perform_initialization:
        init_method(layer.weight)
    with torch.no_grad():
        layer.bias.zero_()
    return layer

@torch.jit.script
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))
def openai_gelu(x):
    return gelu_impl(x)

#This is actually Python equivalent of torch.nn.functional.gelu(), also with type hints for ONNX exporter
@torch.jit.script
def erf_gelu(x):
    return x * 0.5 * (torch.erf(x / 1.41421).to(dtype=x.dtype)+torch.ones_like(x).to(dtype=x.dtype))


def slice_lm_inputs_along_cp(input_ids, position_ids, attention_mask, labels):
    if input_ids is None:  # no data loaded
        return input_ids, position_ids, attention_mask, labels
    CP = mpu.get_context_parallel_world_size()
    if CP >= 2:
        # Check inputs with the same context parallel rank are equal
        args = get_args()
        if args.curr_iteration < args.iteration + args.kaimm_warmup_iters:
            max_input_ids = input_ids.clone()
            torch.distributed.all_reduce(max_input_ids, op=torch.distributed.ReduceOp.MAX,
                                         group=mpu.get_context_parallel_group())
            if (max_input_ids != input_ids).any():
                raise ValueError("Inputs with the same get_data_parallel_for_sample_rank() should be equal. "
                                 "Please check the dataloader.")

        cp_rank = mpu.get_context_parallel_rank()
        input_ids = dattention.slice_cp(input_ids, 1, CP, cp_rank)
        position_ids = dattention.slice_cp(position_ids, 1, CP, cp_rank)
        labels = dattention.slice_cp(labels, 1, CP, cp_rank)
    return input_ids, position_ids, attention_mask, labels


def gather_post_lm_output_along_cp(output):
    return dattention.forward_gather_backward_slice(output, 1, mpu.get_context_parallel_group())
