import contextlib
import math
import os
import threading
import time
from collections import defaultdict
import torch

from megatron.core import parallel_state
from megatron import get_args
_MEMORY_STREAM = dict()
_GPU_BUFFER_POOL = dict()
_CPU_BUFFER_POOL = list()


def get_memcpy_stream(key) -> torch.cuda.Stream:
    """Get a stream for memory copy."""
    if key not in _MEMORY_STREAM:
        _MEMORY_STREAM[key] = torch.cuda.Stream()
    return _MEMORY_STREAM[key]


total_size = []
total_size.append(0.0)

gpu_buffer_lock = threading.Lock()


def get_gpu_buffer(key, size):
    """Get a buffer for GPU."""
    with gpu_buffer_lock:
        if key not in _GPU_BUFFER_POOL or _GPU_BUFFER_POOL[key].numel() < size:
            _GPU_BUFFER_POOL[key] = None
            _GPU_BUFFER_POOL[key] = torch.empty(size, dtype=torch.uint8, device="cuda")
            _GPU_BUFFER_POOL[key].ref_cnt = 0  # ref_cnt supports ping_pong onload
            total_size[0] += size
            if parallel_state.get_pipeline_model_parallel_rank() == 0:
                print(
                    f"rank {parallel_state.get_pipeline_model_parallel_rank()} allocated gpu mem size {total_size[0]/1024/1024} Mib"
                )
        return _GPU_BUFFER_POOL[key][:size]


cpu_buffer_lock = threading.Lock()


def get_cpu_buffer(size):
    """Get a buffer for CPU."""
    with cpu_buffer_lock:
        best_idx = -1
        for i, buffer in enumerate(_CPU_BUFFER_POOL):
            if buffer.numel() >= size:
                if best_idx == -1 or buffer.numel() < _CPU_BUFFER_POOL[best_idx].numel():
                    best_idx = i
        if best_idx != -1:
            return _CPU_BUFFER_POOL.pop(best_idx)[:size]
        if _CPU_BUFFER_POOL:
            _CPU_BUFFER_POOL.pop()
        buffer = torch.empty(size, dtype=torch.int8, device="cpu")
        return buffer[:size]


def recycle_cpu_buffer(buffer):
    """Recycle a CPU buffer."""
    with cpu_buffer_lock:
        _CPU_BUFFER_POOL.append(buffer._base)


class TensorWrap:
    def __init__(self, x: torch.Tensor):
        self.x = x
        self.shape = x.shape
        self.dtype = x.dtype
        self.device = x.device
        self.base = None


class TensorPack:
    def __init__(self, tensor_wrap: TensorWrap):
        self.tensor_wrap = tensor_wrap

    def get(self):
        return self.tensor_wrap.x

    def __del__(self):
        self.tensor_wrap.x = None
        if self.tensor_wrap.base is not None:
            self.tensor_wrap.base.ref_cnt -= 1


def maybe_contiguous(x):
    if x.is_contiguous():
        return x
    else:
        return x.contiguous()


class ActivationGroup:
    def __init__(self, tensors):
        args = get_args()
        self.tensors = [tensor for tensor in tensors if tensor.x is not None]
        self.tensors = sorted(self.tensors, key=lambda t: (not t.x.is_contiguous(), -t.shape.numel()))
        self.offload_ratio = 1.0
        self.offload_size = 1000 * 10 * (2**20)  # bytes
        # if self.offload_ratio > .5:
        #     self.tensors = self.tensors[::-1]

    def offload_prologue(self, use_bucket):
        if not self.tensors:
            return None, None
        self.tensor_to_buffer_map = list()
        top = 0

        for i, tensor in enumerate(self.tensors):
            is_duplicate = False
            if tensor.x == None:
                raise RuntimeError("offloading tensor is None")
            if tensor.x.is_contiguous():
                for j, prev_tensor in enumerate(self.tensors[:i]):
                    if (
                        tensor.x.data_ptr() == prev_tensor.x.data_ptr()
                        and prev_tensor.x.is_contiguous()
                        and tensor.device == prev_tensor.device
                        and tensor.shape.numel() == prev_tensor.shape.numel()
                    ):
                        begin_index, end_index, _ = self.tensor_to_buffer_map[j]
                        is_duplicate = True
                        self.tensor_to_buffer_map.append((begin_index, end_index, is_duplicate))
                        break
            if not is_duplicate:
                n = tensor.shape.numel() * tensor.dtype.itemsize
                self.tensor_to_buffer_map.append((top, top + n, is_duplicate))
                top += n
            if top > self.offload_size:
                break

        MiB = 2**20
        offload_size = min((int(math.ceil(top * self.offload_ratio)) + MiB - 1) // MiB * MiB, self.offload_size)

        if use_bucket:
            buffer = get_gpu_buffer("offload", offload_size)
        else:
            buffer = None

        copy_tasks = []
        partially_offload_bases = set()
        for tensor, (begin_index, end_index, is_duplicate) in zip(self.tensors, self.tensor_to_buffer_map):
            if tensor.x == None:
                raise RuntimeError("offloading tensor is None")
            assert tensor.x.device.type == "cuda"
            if end_index <= offload_size:
                if not is_duplicate:
                    if tensor.x._base is not None:
                        partially_offload_bases.add(tensor.x._base)
                    if use_bucket:
                        buffer[begin_index:end_index].view(tensor.dtype).view(tensor.shape).copy_(tensor.x)
                    else:
                        copy_tasks.append((begin_index, end_index, tensor.x))
                tensor.x = None
            elif begin_index < offload_size:
                if not is_duplicate:
                    if tensor.x._base is not None:
                        partially_offload_bases.add(tensor.x._base)
                    linear_data = maybe_contiguous(tensor.x).view(-1).view(torch.int8)
                    if use_bucket:
                        buffer[begin_index:].copy_(linear_data[: offload_size - begin_index])
                    else:
                        copy_tasks.append((begin_index, offload_size, linear_data[: offload_size - begin_index]))
                    self.remained_not_offload = linear_data[offload_size - begin_index :].clone()
                tensor.x = None
            elif tensor.x._base in partially_offload_bases:
                if is_duplicate:
                    raise NotImplementedError("Does not support partially offload duplicate tensors")
                tensor.x = tensor.x.clone()

        self.cpu_buffer = get_cpu_buffer(offload_size)
        stream = get_memcpy_stream("offload")
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            if use_bucket:
                self.cpu_buffer.copy_(buffer, non_blocking=False)
            else:
                for begin_index, end_index, x in copy_tasks:
                    self.cpu_buffer[begin_index:end_index].view(x.dtype).view(x.shape).copy_(x, non_blocking=False)

        return stream, buffer

    def offload_epilogue(self, stream, buffer):
        if not self.tensors:
            return
        torch.cuda.current_stream().wait_stream(stream)

    def onload_prologue(self, *, overlap_d2h_h2d, ping_pong_onload):
        if not self.tensors:
            return None, None, ping_pong_onload
        stream_key = "onload" if overlap_d2h_h2d else "offload"
        if ping_pong_onload:
            buffer_key = "onload_ping"
            buffer = get_gpu_buffer(buffer_key, self.cpu_buffer.numel())
            if buffer._base.ref_cnt > 0:
                buffer_key = "onload_pong"
            buffer = get_gpu_buffer(buffer_key, self.cpu_buffer.numel())
            if buffer._base.ref_cnt > 0:
                buffer_key = "onload_three"
                buffer = get_gpu_buffer(buffer_key, self.cpu_buffer.numel())
        else:
            buffer_key = stream_key
        stream = get_memcpy_stream(stream_key)
        buffer = get_gpu_buffer(buffer_key, self.cpu_buffer.numel())
        assert buffer._base.ref_cnt == 0, "last onload tensors are not fully deleted"
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            buffer.copy_(self.cpu_buffer, non_blocking=True)
        return stream, buffer, ping_pong_onload

    def onload_epilogue(self, stream, buffer, ping_pong_onload):
        if not self.tensors:
            return None
        torch.cuda.current_stream().wait_stream(stream)
        recycle_cpu_buffer(self.cpu_buffer)
        self.cpu_buffer = None
        offload_size = buffer.numel()
        duplicate_tensors = dict()
        for tensor, (begin_idx, end_idx, duplicate_flag) in zip(self.tensors, self.tensor_to_buffer_map):
            if end_idx <= offload_size:
                tensor.x = buffer[begin_idx:end_idx].view(tensor.dtype).view(tensor.shape)
                if ping_pong_onload:
                    tensor.base = buffer._base
                    tensor.base.ref_cnt += 1
                else:
                    tensor.x = tensor.x.clone()
            elif begin_idx < offload_size:
                if not duplicate_flag:
                    tensor.x = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
                    linear_data = tensor.x.view(-1).view(buffer.dtype)
                    linear_data[: offload_size - begin_idx].copy_(buffer[begin_idx:])
                    linear_data[offload_size - begin_idx :].copy_(self.remained_not_offload)
                    self.remained_not_offload = None
                    duplicate_tensors[begin_idx, end_idx] = linear_data
                else:
                    tensor.x = duplicate_tensors[begin_idx, end_idx].view(tensor.dtype).view(tensor.shape)
        del self.tensors
        del self.tensor_to_buffer_map


groups = dict()

total_none = [0]


@contextlib.contextmanager
def fake_record(key):
    groups[key] = ActivationGroup([])

    def pack_hook(x):
        tensor_wrap = TensorWrap(x)
        return TensorPack(tensor_wrap)

    def unpack_hook(tensor_pack):
        return tensor_pack.get()

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        yield


class Args:
    selective_recompute_offload_transformer_layer = True

# def get_args():
#     """Get the arguments."""
#     return Args()
@contextlib.contextmanager
def record(key):
    args = get_args()
    if not args.selective_recompute_offload_transformer_layer:
        yield
        groups[key] = ActivationGroup([])
        return

    tensors = list()

    def pack_hook(x):
        tensor_wrap = TensorWrap(x)
        is_parameter = isinstance(x, torch.nn.Parameter)
        is_too_small = x.numel() * x.element_size() < 1024 * 1024 * 1
        is_rope_freqs = x.dim() == 4 and x.shape[1] == 1 and x.shape[2] == 1
        is_attention_activation = x.dim() == 3 and(x.grad_fn == None or type(x.grad_fn).__name__ == "FlashAttnVarlenFuncBackward" or type(x.grad_fn).__name__ == "ViewBackward0")
        # is_rmsnorm = type(x.grad_fn).__name__ == "FusedLayerNormAffineFunctionBackward" or type(x.grad_fn).__name__ == "FusedRMSNormFunctionBackward"
        # print(f"record x address {hex(x.data_ptr())} shape {x.shape} grad_fn {x.grad_fn}")
        # is_input = type(x.grad_fn).__name__ == "WrapInputsFunctionBackward"

        if not is_parameter and not is_too_small and not is_rope_freqs and (is_attention_activation ):
            # or is_rmsnorm
            # print(f"offload x address {hex(x.data_ptr())} shape {x.shape} grad_fn {x.grad_fn}")

            # if type(x.grad_fn).__name__ == "FusedLayerNormAffineFunctionBackward":
            #     print(f"FusedLayerNormAffineFunctionBackward x address {hex(x.data_ptr())} shape {x.shape} grad_fn {x.grad_fn}")
            tensors.append(tensor_wrap)
        return TensorPack(tensor_wrap)

    def unpack_hook(tensor_pack: TensorPack):
        x = tensor_pack.get()
        return x

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        yield

    groups[key] = ActivationGroup(tensors)


@contextlib.contextmanager
def offload_async(key):
    group = groups[key]

    args_container = []

    def prologue_thread():
        # time.sleep(0)
        # torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        args_container.append(group.offload_prologue(use_bucket=False))

    thread = threading.Thread(target=prologue_thread)
    thread.start()
    yield
    thread.join()
    group.offload_epilogue(*args_container[0])


@contextlib.contextmanager
def onload_async(key):
    group = groups[key]
    args_container = []

    def prologue_thread():
        # torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        args_container.append(group.onload_prologue(overlap_d2h_h2d=True, ping_pong_onload=True))
        group.onload_epilogue(*args_container[0])

    thread = threading.Thread(target=prologue_thread)
    thread.start()
    yield
    thread.join()
    


# @contextlib.contextmanager
# def onload_async(key):
#     group = groups[key]
#     # start = time.time()
#     args = group.onload_prologue(overlap_d2h_h2d=True, ping_pong_onload=True)
#     # print(f"onload_prologue time {(time.time() - start) * 1000} s")
#     yield
#     # start = time.time()
#     group.onload_epilogue(*args)
#     # print(f"onload_epilogue time {(time.time() - start) * 1000} s")