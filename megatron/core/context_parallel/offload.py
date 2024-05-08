# Copyright (c) 2024, Kuaishou Technology. All rights reserved.

import torch

from collections import defaultdict


_PINNED_BUFFER_POOL = defaultdict(list)
_MEMCPY_STREAM = None


def set_ideal_affinity_for_current_gpu():
    import cuda.cuda
    import cuda.cudart
    import pynvml
    import uuid
    err, device_id = cuda.cudart.cudaGetDevice()
    assert err == cuda.cudart.cudaError_t.cudaSuccess
    err, device_uuid = cuda.cuda.cuDeviceGetUuid(device_id)
    assert err == cuda.cuda.CUresult.CUDA_SUCCESS
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByUUID("GPU-" + str(uuid.UUID(bytes=device_uuid.bytes)))
    pynvml.nvmlDeviceSetCpuAffinity(handle)


def get_memcpy_stream():
    global _MEMCPY_STREAM
    if _MEMCPY_STREAM is None:
        _MEMCPY_STREAM = torch.cuda.Stream()
    return _MEMCPY_STREAM


def get_pinned_buffer_from_pool(num_bytes, dtype=torch.uint8):
    if not _PINNED_BUFFER_POOL[num_bytes, dtype]:
        set_ideal_affinity_for_current_gpu()
        _PINNED_BUFFER_POOL[num_bytes, dtype].append(torch.empty(num_bytes, dtype=dtype, pin_memory=True))
    return _PINNED_BUFFER_POOL[num_bytes, dtype].pop()


def recycle_pinned_buffer(buffer):
    _PINNED_BUFFER_POOL[buffer.numel(), buffer.dtype].append(buffer)


class LoadAsyncHandle:
    def __init__(self, pinned_buffer, memcpy_h2d_event):
        self.pinned_buffer = pinned_buffer
        self.memcpy_h2d_event = memcpy_h2d_event

    def wait(self):
        self.memcpy_h2d_event.wait()
        if self.pinned_buffer is not None:
            recycle_pinned_buffer(self.pinned_buffer)
            self.pinned_buffer = None


class OffloadPhase1Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, data_to_save, group):
        assert data_to_save.is_contiguous()
        ctx.shape = data_to_save.shape
        world_size = torch.distributed.get_world_size(group)
        rank = torch.distributed.get_rank(group)
        data_pinned = get_pinned_buffer_from_pool(data_to_save.numel() // world_size, data_to_save.dtype)
        get_memcpy_stream().wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(get_memcpy_stream()):
            data_pinned.copy_(data_to_save.view(world_size, -1)[rank], non_blocking=True)
            data_to_save.record_stream(torch.cuda.current_stream())
        return x, data_to_save, data_pinned

    @staticmethod
    def backward(ctx, grad_output, saved_data, data_pinned):
        return grad_output, saved_data, None


class OffloadPhase2Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, data_to_save, data_pinned, group):
        ctx.group = group
        return x, data_to_save, data_pinned

    @staticmethod
    def backward(ctx, grad_output, saved_data, data_pinned):
        world_size = torch.distributed.get_world_size(ctx.group)
        if world_size >= 2:
            rank = torch.distributed.get_rank(ctx.group)
            saved_data._handle.wait()
            saved_data._handle = torch.distributed.all_gather_into_tensor(
                saved_data, saved_data.view(world_size, -1)[rank], group=ctx.group, async_op=True)
        return grad_output, saved_data, data_pinned, None


class OffloadPhase3Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, data_to_save, data_pinned, group):
        ctx.group = group
        ctx.shape = data_to_save.shape
        ctx.save_for_backward(data_pinned)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        world_size = torch.distributed.get_world_size(ctx.group)
        rank = torch.distributed.get_rank(ctx.group)
        data_pinned, = ctx.saved_tensors
        saved_data = torch.empty(ctx.shape, dtype=data_pinned.dtype, device="cuda")
        get_memcpy_stream().wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(get_memcpy_stream()):
            saved_data.view(world_size, -1)[rank].copy_(data_pinned, non_blocking=True)
            saved_data.record_stream(torch.cuda.current_stream())
            memcpy_h2d_event = torch.cuda.Event()
            memcpy_h2d_event.record()
        saved_data._handle = LoadAsyncHandle(data_pinned, memcpy_h2d_event)
        return grad_output, saved_data, data_pinned, None


def offload_phase1(x, data_to_save, group):
    return *OffloadPhase1Function.apply(x, data_to_save, group), group


def offload_phase2(x, *offload_phase2_args):
    group = offload_phase2_args[-1]
    return *OffloadPhase2Function.apply(x, *offload_phase2_args), group


def offload_phase3(x, *offload_phase3_args):
    return OffloadPhase3Function.apply(x, *offload_phase3_args)


def wait_offload_stream():
    torch.cuda.current_stream().wait_stream(get_memcpy_stream())
