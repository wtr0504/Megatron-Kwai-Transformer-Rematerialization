# Copyright (c) 2024, Kuaishou Technology. All rights reserved.

import functools
import pathlib
import torch

from .dispatch_flash_attn import flash_attn_func, _flash_attn_forward, _flash_attn_backward


# example before flip, rank 0: [0 1 2 3],   rank 1: [4 5 6 7],   rank 2: [8 9 10 11], rank 3: [12 13 14 15]
#          after flip, rank 0: [0 1 14 15], rank 1: [4 5 10 11], rank 2: [8 9 6 7],   rank 3: [12 13 2 3]

def flip_cp_(x, dim, world_size):
    if world_size == 1:
        return x
    batch_size = x.shape[:dim].numel()
    v = x.view(batch_size, world_size, 2, -1)[:, :, 1]

    # Fast v.copy_(v.flip(1))
    import fast_flip_cuda
    assert v.device.type == "cuda", "the fused op only supports CUDA"
    assert v.stride(2) == 1, "the fused op requires the last dim to be contiguous"
    fast_flip_cuda.flip(v.data_ptr(), v.data_ptr(), v.shape[0], v.stride(0), v.shape[1], v.stride(1), v.shape[2], v.element_size(), torch.cuda.current_stream().cuda_stream)

    return x


def flip_cp(x, dim, world_size):
    if world_size == 1:
        return x
    o = torch.empty_like(x)
    vx = x.view(*x.shape[:dim], world_size, 2, x.shape[dim] // world_size // 2, *x.shape[dim + 1:])
    vo = o.view(*x.shape[:dim], world_size, 2, x.shape[dim] // world_size // 2, *x.shape[dim + 1:])
    vo.select(dim + 1, 0).copy_(vx.select(dim + 1, 0))
    vo.select(dim + 1, 1).copy_(vx.select(dim + 1, 1).flip(dim))
    return o


@functools.lru_cache
def _get_index_select_index(world_size, rank, device):
    return torch.tensor([rank * 2, 2 * world_size - 1 - 2 * rank], device=device)


def slice_cp(x, dim, world_size, rank):
    if world_size == 1:
        assert rank == 0
        return x
    vx = x.view(*x.shape[:dim], world_size * 2, x.shape[dim] // world_size // 2, *x.shape[dim + 1:])
    output = vx.index_select(dim, _get_index_select_index(world_size, rank, vx.device))
    return output.view(*x.shape[:dim], x.shape[dim] // world_size, *x.shape[dim + 1:])


def all_gather_along_dim(input, dim, group):
    world_size = torch.distributed.get_world_size(group)
    output = torch.empty(world_size, *input.shape, dtype=input.dtype, device=input.device)
    torch.distributed.all_gather_into_tensor(output, input.contiguous(), group=group)
    output = output.permute(*range(1, dim + 1), 0, *range(dim + 1, input.dim() + 1))
    output = output.reshape(*input.shape[:dim], world_size * input.shape[dim], *input.shape[dim + 1:])
    return output


def reduce_scatter_along_dim(input, dim, group):
    world_size = torch.distributed.get_world_size(group)
    output = torch.empty(input.shape[dim] // world_size, *input.shape[:dim], *input.shape[dim + 1:], dtype=input.dtype, device=input.device)
    torch.distributed.reduce_scatter_tensor(output, input.permute(dim, *range(dim), *range(dim + 1, input.dim())).contiguous(), group=group)
    output = output.permute(*range(1, dim + 1), 0, *range(dim + 1, input.dim()))
    return output


_CP_STREAM = None


def get_cp_stream():
    global _CP_STREAM
    if _CP_STREAM is None:
        _CP_STREAM = torch.cuda.Stream()
    return _CP_STREAM


# The qi refers to the i-th shard of q.
# The qi is sharded along the second axis (the seqlen axis).
# The kvT refers to kv.transpose(0, 1) whose shape is (s, b, 2, num_heads, head_dim).
# The kvTi is sharded along the first axis (the seqlen axis).
# Sharding on kvT (instead of kv) helps to avoid transpose before NCCL communication.
# Flip seqlen (call flip_cp_) before shard tensors.


class DAttentionPreFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ki, vi, cp_group):
        ctx.cp_group = cp_group
        kvTi = torch.stack([ki.transpose(0, 1), vi.transpose(0, 1)], dim=2)
        kvT = all_gather_along_dim(kvTi, 0, group=ctx.cp_group)
        flip_cp_(kvT, 0, torch.distributed.get_world_size(ctx.cp_group))
        kv = kvT.transpose(0, 1)
        return kv

    @staticmethod
    def backward(ctx, grad_kv):
        grad_kvT = grad_kv.transpose(0, 1)
        flip_cp_(grad_kvT, 0, torch.distributed.get_world_size(ctx.cp_group))
        grad_kvTi = reduce_scatter_along_dim(grad_kvT, 0, group=ctx.cp_group)
        # grad_kvTi = reduce_scatter_along_dim(grad_kvT.float(), 0, group=ctx.cp_group).to(grad_kv.dtype)
        grad_kvi = grad_kvTi.transpose(0, 1)
        return grad_kvi[:, :, 0], grad_kvi[:, :, 1], None


class DAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qi, kv, cp_group):
        ctx.cp_group = cp_group
        CP = torch.distributed.get_world_size(ctx.cp_group)
        cp_rank = torch.distributed.get_rank(ctx.cp_group)
        b, seqlen_qi, a, d = qi.shape
        seqlen_kv = kv.shape[1]

        if kv.transpose(0, 1).is_contiguous():
            data_to_save = kv.transpose(0, 1)
            ctx.convert_saved_data_to_kv = lambda x: x.transpose(0, 1)
        elif kv.permute(2, 1, 0, 3, 4).is_contiguous():
            data_to_save = kv.permute(2, 1, 0, 3, 4)
            ctx.convert_saved_data_to_kv = lambda x: x.permute(2, 1, 0, 3, 4)
        else:
            raise NotImplementedError(f"unsupported kv shape {kv.shape} stride {kv.stride()}")

        ctx.use_sdp = False
        if ctx.use_sdp:
            attn_bias = torch.full((seqlen_qi, seqlen_kv), float("-inf"), dtype=qi.dtype, device=qi.device)
            attn_bias[:seqlen_qi // 2].triu_(1 + cp_rank * seqlen_kv // CP)
            attn_bias[seqlen_qi // 2:].triu_(1 + (2 * CP - 1 - 2 * cp_rank) * seqlen_kv // CP // 2)
            attn_bias = attn_bias.expand(b, a, *attn_bias.shape)
            compute_log_sumexp = True
            dropout_p = 0.
            is_causal = False
            output, log_sumexp, philox_seed, philox_offset = \
                torch.ops.aten._scaled_dot_product_efficient_attention(
                    qi.transpose(1, 2), kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2),
                    attn_bias, compute_log_sumexp, dropout_p, is_causal)
            oi = output.transpose(1, 2)

            ctx.save_for_backward(qi, log_sumexp, attn_bias, oi, log_sumexp, philox_seed, philox_offset)
            ctx.dropout_p = dropout_p
            ctx.is_causal = is_causal
            return oi, data_to_save

        dropout_p = 0.
        softmax_scale = d ** -.5
        causal = True
        window_size = (-1, -1)
        return_softmax = False

        qi0 = qi[:, :seqlen_qi // 2]
        kv0 = kv[:, :(2 * cp_rank + 1) * seqlen_kv // (2 * CP)]

        qi1 = qi[:, seqlen_qi // 2:]
        kv1 = kv[:, :(CP - cp_rank) * seqlen_kv // CP]

        oi0, qi_padded0, k_padded0, v_padded0, out_padded0, softmax_lse0, S_dmask0, rng_state0 = (None,) * 8
        oi1, qi_padded1, k_padded1, v_padded1, out_padded1, softmax_lse1, S_dmask1, rng_state1 = (None,) * 8

        def attn_func0():
            nonlocal oi0, qi_padded0, k_padded0, v_padded0, out_padded0, softmax_lse0, S_dmask0, rng_state0
            oi0, qi_padded0, k_padded0, v_padded0, out_padded0, softmax_lse0, S_dmask0, rng_state0 = _flash_attn_forward(
                qi0,
                kv0[:, :, 0],
                kv0[:, :, 1],
                dropout_p,
                softmax_scale,
                causal=causal,
                window_size=window_size,
                return_softmax=return_softmax and dropout_p > 0,
            )
            assert (qi0.shape, kv0[:, :, 0].shape, kv0[:, :, 1].shape) == (qi_padded0.shape, k_padded0.shape, v_padded0.shape), "no support padding"
            assert (oi0.data_ptr(), oi0.shape, oi0.stride()) == (out_padded0.data_ptr(), out_padded0.shape, out_padded0.stride()), "no support padding"

        def attn_func1():
            nonlocal oi1, qi_padded1, k_padded1, v_padded1, out_padded1, softmax_lse1, S_dmask1, rng_state1
            oi1, qi_padded1, k_padded1, v_padded1, out_padded1, softmax_lse1, S_dmask1, rng_state1 = _flash_attn_forward(
                qi1,
                kv1[:, :, 0],
                kv1[:, :, 1],
                dropout_p,
                softmax_scale,
                causal=causal,
                window_size=window_size,
                return_softmax=return_softmax and dropout_p > 0,
            )
            assert (qi1.shape, kv1[:, :, 0].shape, kv1[:, :, 1].shape) == (qi_padded1.shape, k_padded1.shape, v_padded1.shape), "no support padding"
            assert (oi1.data_ptr(), oi1.shape, oi1.stride()) == (out_padded1.data_ptr(), out_padded1.shape, out_padded1.stride()), "no support padding"

        get_cp_stream().wait_stream(torch.cuda.current_stream())
        if kv0.shape[1] >= kv1.shape[1]:  # call the longer kernel first
            attn_func0()
            with torch.cuda.stream(get_cp_stream()):
                attn_func1()
        else:
            with torch.cuda.stream(get_cp_stream()):
                attn_func1()
            attn_func0()
        torch.cuda.current_stream().wait_stream(get_cp_stream())

        oi = torch.concat([oi0, oi1], dim=1)
        ctx.save_for_backward(qi, oi, softmax_lse0, rng_state0, softmax_lse1, rng_state1)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size

        return oi, data_to_save

    @staticmethod
    def backward(ctx, grad_oi, saved_data):
        CP = torch.distributed.get_world_size(ctx.cp_group)
        cp_rank = torch.distributed.get_rank(ctx.cp_group)
        saved_data._handle.wait()
        del saved_data._handle  # break circular reference
        kv = ctx.convert_saved_data_to_kv(saved_data)

        if ctx.use_sdp:
            qi, log_sumexp, attn_bias, oi, log_sumexp, philox_seed, philox_offset = ctx.saved_tensors
            b, seqlen_qi, a, d = qi.shape
            seqlen_kv = seqlen_qi * CP

            grad_input_mask = ctx.needs_input_grad[:3] + (False,)
            grad_qi, grad_k, grad_v, grad_bias = torch.ops.aten._scaled_dot_product_efficient_attention_backward(
                grad_oi.transpose(1, 2), qi.transpose(1, 2), kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2), attn_bias, oi.transpose(1, 2),
                log_sumexp, philox_seed, philox_offset, ctx.dropout_p, grad_input_mask, ctx.is_causal)
            grad_qi, grad_k, grad_v = grad_qi.transpose(1, 2), grad_k.transpose(1, 2), grad_v.transpose(1, 2)
            grad_kv = torch.empty_strided(kv.shape, kv.stride(), dtype=kv.dtype, device=kv.device)
            grad_kv[:, :, 0] = grad_k
            grad_kv[:, :, 1] = grad_v
            return grad_qi, grad_kv, None, None, None, None

        qi, oi, softmax_lse0, rng_state0, softmax_lse1, rng_state1 = ctx.saved_tensors
        out_padded0, out_padded1 = oi.chunk(2, dim=1)
        b, seqlen_qi, a, d = qi.shape
        seqlen_kv = seqlen_qi * CP

        dqi = torch.empty_like(qi)
        dkv = torch.empty_strided(kv.shape, kv.stride(), dtype=kv.dtype, device=kv.device)

        qi0 = qi[:, :seqlen_qi // 2]
        kv0 = kv[:, :(2 * cp_rank + 1) * seqlen_kv // (2 * CP)]
        doi0 = grad_oi[:, :seqlen_qi // 2]
        dqi0 = dqi[:, :seqlen_qi // 2]

        qi1 = qi[:, seqlen_qi // 2:]
        kv1 = kv[:, :(CP - cp_rank) * seqlen_kv // CP]
        doi1 = grad_oi[:, seqlen_qi // 2:]
        dqi1 = dqi[:, seqlen_qi // 2:]

        kv0_is_longer = kv0.shape[1] >= kv1.shape[1]
        if kv0_is_longer:
            dkv0 = dkv[:, :kv0.shape[1]]
            dkv1 = torch.empty_like(kv1)
        else:
            dkv0 = torch.empty_like(kv0)
            dkv1 = dkv[:, :kv1.shape[1]]

        get_cp_stream().wait_stream(torch.cuda.current_stream())
        _flash_attn_backward(
            doi0,
            qi0,
            kv0[:, :, 0],
            kv0[:, :, 1],
            out_padded0,
            softmax_lse0,
            dqi0,
            dkv0[:, :, 0],
            dkv0[:, :, 1],
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size,
            rng_state=rng_state0,
        )
        with torch.cuda.stream(get_cp_stream()):
            _flash_attn_backward(
                doi1,
                qi1,
                kv1[:, :, 0],
                kv1[:, :, 1],
                out_padded1,
                softmax_lse1,
                dqi1,
                dkv1[:, :, 0],
                dkv1[:, :, 1],
                ctx.dropout_p,
                ctx.softmax_scale,
                ctx.causal,
                ctx.window_size,
                rng_state=rng_state1,
            )
        torch.cuda.current_stream().wait_stream(get_cp_stream())

        if kv0_is_longer:
            dkv[:, :dkv1.shape[1]] += dkv1
        else:
            dkv[:, :dkv0.shape[1]] += dkv0
        dkv[:, max(dkv0.shape[1], dkv1.shape[1]):] = 0
        return dqi, dkv, None, None, None, None


class ShardSaveForBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, data_to_save, group):
        ctx.group = group
        ctx.world_size = torch.distributed.get_world_size(group)
        ctx.rank = torch.distributed.get_rank(group)
        assert data_to_save.is_contiguous()
        data_shard = data_to_save.view(ctx.world_size, -1)[ctx.rank].clone()
        ctx.shape = data_to_save.shape
        ctx.save_for_backward(data_shard)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        data_shard, = ctx.saved_tensors
        saved_data = torch.empty(ctx.shape, dtype=data_shard.dtype, device=data_shard.device)
        saved_data._handle = torch.distributed.all_gather_into_tensor(saved_data, data_shard, group=ctx.group, async_op=True)
        return grad_output, saved_data, None


class FlipInplaceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, CP):
        ctx.CP = CP
        # Convert the argument of flip_cp_ to continuous layout.
        # Refer to dattention_overlap for KV layout.
        kv_2sbad = kv.permute(2, 1, 0, 3, 4)
        flip_cp_(kv_2sbad, 1, CP)
        return kv

    @staticmethod
    def backward(ctx, dkv):
        dkv_2sbad = dkv.permute(2, 1, 0, 3, 4)
        flip_cp_(dkv_2sbad, 1, ctx.CP)
        return dkv, None


class ForwardGatherBackwardSliceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, cp_group):
        CP = torch.distributed.get_world_size(cp_group)
        cp_rank = torch.distributed.get_rank(cp_group)
        ctx.dim = dim
        ctx.CP = CP
        ctx.cp_rank = cp_rank
        x = all_gather_along_dim(x, dim, cp_group)
        flip_cp_(x, dim, CP)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return slice_cp(grad_output, ctx.dim, ctx.CP, ctx.cp_rank), None, None


def dattention(qi, ki, vi, cp_group):
    if torch.distributed.get_world_size(cp_group) == 1:
        return flash_attn_func(qi, ki, vi, causal=True)
    kv = DAttentionPreFunction.apply(ki, vi, cp_group)
    oi, data_to_save = DAttentionFunction.apply(qi, kv, cp_group)
    return oi, data_to_save


def dattention_overlap(qi, kv_2sbad, cp_group):
    """The layout of kv_2sbad is (2, s, b, num_heads, head_dim).
    This is the native layout after gathering V and K respectively.
    """
    CP = torch.distributed.get_world_size(cp_group)
    assert CP >= 2, "dattention overlap is not optimized for CP=1"
    kv = kv_2sbad.permute(2, 1, 0, 3, 4)
    kv = FlipInplaceFunction.apply(kv, CP)
    oi, data_to_save = DAttentionFunction.apply(qi, kv, cp_group)
    return oi, data_to_save


def shard_save_for_backward(x, data_to_save, group):
    return ShardSaveForBackwardFunction.apply(x, data_to_save, group)


def forward_gather_backward_slice(x, dim, cp_group):
    return ForwardGatherBackwardSliceFunction.apply(x, dim, cp_group)
