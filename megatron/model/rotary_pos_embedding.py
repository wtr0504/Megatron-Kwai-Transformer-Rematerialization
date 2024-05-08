# coding=utf-8

# The following code has been taken from https://github.com/NVIDIA/NeMo/blob/ \
# 782b4e1652aaa43c8be390d9db0dc89544afa080/nemo/collections/nlp/modules/ \
# common/megatron/rotary_pos_embedding.py

import functools
import importlib.util
import torch

from megatron.core.context_parallel import dattention
from torch import einsum, nn

__all__ = ['RotaryEmbedding', 'apply_rotary_pos_emb']

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, use_fast_rope=False, context_parallel_world_size=1, context_parallel_rank=0):
        super().__init__()
        self.dim = dim
        self.use_fast_rope = use_fast_rope
        self.context_parallel_world_size = context_parallel_world_size
        self.context_parallel_rank = context_parallel_rank
        self.register_buffer('dummy_buffer', torch.tensor(1.))
        if importlib.util.find_spec('einops') is None:
            raise RuntimeError("einops is required for Rotary Embedding")
        self.forward = functools.lru_cache(maxsize=1)(self.forward)

    def forward(self, max_seq_len, offset=0):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2, device=self.dummy_buffer.device).float() / self.dim))
        seq = torch.arange(max_seq_len, device=inv_freq.device) + offset
        seq = dattention.slice_cp(seq, 0, self.context_parallel_world_size, self.context_parallel_rank)
        freqs = einsum('i , j -> i j', seq.type_as(inv_freq), inv_freq)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        emb = torch.cat((freqs, freqs), dim=-1)
        # emb [seq_length, .., dim]
        from einops import rearrange
        freqs = rearrange(emb, 'n d -> n 1 1 d')
        if self.use_fast_rope:
            freqs_sin_cos = torch.cat([freqs.sin()[..., None], freqs.cos()[..., None]], dim=-1).reshape(*freqs.shape[:-1], -1)
            return freqs_sin_cos.type_as(self.dummy_buffer)
        # Note(yuantailing): Store freqs (before sin/cos) in fp32 to match fast rope precision
        return freqs


class FastRotaryPosEmbFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, freqs, precompute_sin_cos):
        import fast_rotary_pos_emb
        output = fast_rotary_pos_emb.forward(t, freqs, precompute_sin_cos)
        ctx.save_for_backward(freqs)
        ctx.precompute_sin_cos = precompute_sin_cos
        return output

    @staticmethod
    def backward(ctx, grad_output):
        import fast_rotary_pos_emb
        freqs, = ctx.saved_tensors
        d_t = fast_rotary_pos_emb.backward(grad_output, freqs, ctx.precompute_sin_cos)
        return d_t, None, None


def _rotate_half(x):
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    from einops import rearrange
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs, use_fast_rope=False):
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
    check https://kexue.fm/archives/8265 for detailed formulas
    """
    if use_fast_rope:
        return FastRotaryPosEmbFunction.apply(t, freqs, True)

    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    # Note(yuantailing): Calculate sin/cos in fp32 to match fast rope precision
    t = (t * freqs.cos().to(t.dtype)) + (_rotate_half(t) * freqs.sin().to(t.dtype))
    return torch.cat((t, t_pass), dim=-1)
