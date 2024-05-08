import math

import torch
from torch.nn import LayerNorm

from megatron.model.enums import AttnMaskType
from megatron.model.fused_layer_norm import MixedFusedLayerNorm
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.utils import attention_mask_func
from megatron.fused_kernels import load

def test_load_fused_kernels():
    try:
        import fused_layer_norm_cuda
        import scaled_masked_softmax_cuda
        import scaled_upper_triang_masked_softmax_cuda
        import torch

        print("[Success] load_fused_kernels")
    except ImportError as e:
        print("[Fail] load_fused_kernels")
        raise e

def test_fused_softmax():
    bert = BertModel.from_pretrained("bert-base-cased").cuda().half()
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    test_text = (
        "Hello. How are you? I am fine thank you and you? yes Good. "
        "hi hi hi hi hi hi hi hi hi hi hi hi hi"  # 32
    )

    tokens = tokenizer(
        [test_text] * 4,
        return_tensors="pt",
    )

    embedding_output = bert.embeddings(
        input_ids=tokens["input_ids"].cuda(),
        position_ids=None,
        token_type_ids=tokens["token_type_ids"].cuda(),
        inputs_embeds=None,
        past_key_values_length=0,
    )

    # (bsz, 1, 1, seq_len)
    mask = bert.get_extended_attention_mask(
        attention_mask=tokens["attention_mask"].cuda(),
        input_shape=tokens["input_ids"].shape,
        device=bert.device,
    )
    # (bsz, 1, seq_len, seq_len)
    mask = mask.repeat(1, 1, mask.size()[-1], 1)

    attention = bert.encoder.layer[0].attention.self
    key_layer = attention.transpose_for_scores(attention.key(embedding_output))
    query_layer = attention.transpose_for_scores(attention.query(embedding_output))

    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores /= math.sqrt(key_layer.size()[-1])

    fused_softmax = (
        FusedScaleMaskSoftmax(
            input_in_fp16=True,
            input_in_bf16=False,
            mask_func=attention_mask_func,
            scale=None,
            softmax_in_fp32=False,
            attn_mask_type=AttnMaskType.padding,
            scaled_masked_softmax_fusion=True,
        )
        .cuda()
        .half()
    )

    fused_softmax_output = fused_softmax(
        attention_scores,
        (mask != 0),
    )

    torch_softmax = (
        FusedScaleMaskSoftmax(
            input_in_fp16=True,
            input_in_bf16=False,
            mask_func=attention_mask_func,
            scale=None,
            softmax_in_fp32=False,
            attn_mask_type=AttnMaskType.padding,
            scaled_masked_softmax_fusion=False,
        )
        .cuda()
        .half()
    )

    torch_softmax_output = torch_softmax(
        attention_scores,
        (mask != 0),
    )

    test_result = (fused_softmax_output - torch_softmax_output).abs()

    while test_result.dim() != 1:
        test_result = test_result.mean(dim=-1)

    diff = test_result.mean(dim=-1)

    if diff <= 1e-3:
        print(
            f"\n[Success] test_fused_softmax"
            f"\n > mean_difference={diff}"
            f"\n > fused_values={fused_softmax_output[-1][-1][-1][:5].tolist()}"
            f"\n > torch_values={torch_softmax_output[-1][-1][-1][:5].tolist()}"
        )
    else:
        print(
            f"\n[Fail] test_fused_softmax"
            f"\n > mean_difference={diff}, "
            f"\n > fused_values={fused_softmax_output[-1][-1][-1][:5].tolist()}, "
            f"\n > torch_values={torch_softmax_output[-1][-1][-1][:5].tolist()}"
        )


def test_fused_upper_triangle_mask_softmax():
    gpt = GPT2Model.from_pretrained("gpt2").cuda().half()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    test_text = (
        "Hello. How are you? I am fine thank you and you? yes Good. "
        "hi hi hi hi hi hi hi"  # 24
    )

    tokens = tokenizer(
        [test_text] * 4,
        return_tensors="pt",
    )

    attention_mask = tokens["attention_mask"].cuda()
    attention_mask = attention_mask.view(attention_mask.size(0), -1)
    attention_mask = attention_mask[:, None, None, :]
    attention_mask = (1.0 - attention_mask) * -10000.0
    attention_mask = attention_mask.repeat(1, 1, attention_mask.size()[-1], 1)
    attn = gpt.h[0]

    hidden_states = gpt.wte(tokens["input_ids"].cuda())
    q, k, v = attn.attn.c_attn(hidden_states).split(768, dim=-1)
    q = attn.attn._split_heads(q, attn.attn.num_heads, attn.attn.head_dim)
    k = attn.attn._split_heads(k, attn.attn.num_heads, attn.attn.head_dim)
    attn_weights = torch.matmul(q, k.transpose(-1, -2))

    sq, sk = q.size(-2), k.size(-2)
    causal_mask = attn.attn.bias[:, :, sk - sq : sk, :sk].bool()
    total_mask = ~(causal_mask & (attention_mask == 0))
    """
    tensor([[[[False,  True,  True,  ...,  True,  True,  True],
              [False, False,  True,  ...,  True,  True,  True],
              [False, False, False,  ...,  True,  True,  True],
              ...,
              [False, False, False,  ..., False,  True,  True],
              [False, False, False,  ..., False, False,  True],
              [False, False, False,  ..., False, False, False]]]
    """

    fused_softmax = (
        FusedScaleMaskSoftmax(
            input_in_fp16=True,
            input_in_bf16=False,
            mask_func=attention_mask_func,
            scale=None,
            softmax_in_fp32=False,
            attn_mask_type=AttnMaskType.causal,
            scaled_masked_softmax_fusion=True,
        )
        .cuda()
        .half()
    )

    fused_softmax_output = fused_softmax(
        attn_weights,
        total_mask,
    )

    torch_softmax = (
        FusedScaleMaskSoftmax(
            input_in_fp16=True,
            input_in_bf16=False,
            mask_func=attention_mask_func,
            scale=None,
            softmax_in_fp32=False,
            attn_mask_type=AttnMaskType.causal,
            scaled_masked_softmax_fusion=False,
        )
        .cuda()
        .half()
    )

    torch_softmax_output = torch_softmax(
        attn_weights,
        total_mask,
    )

    test_result = (fused_softmax_output - torch_softmax_output).abs()

    while test_result.dim() != 1:
        test_result = test_result.mean(dim=-1)

    diff = test_result.mean(dim=-1)

    if diff <= 1e-3:
        print(
            f"\n[Success] test_fused_upper_triangle_mask_softmax"
            f"\n > mean_difference={diff}"
            f"\n > fused_values={fused_softmax_output[-1][-1][-1][:5].tolist()}"
            f"\n > torch_values={torch_softmax_output[-1][-1][-1][:5].tolist()}"
        )
    else:
        print(
            f"\n[Fail] test_fused_upper_triangle_mask_softmax"
            f"\n > mean_difference={diff}, "
            f"\n > fused_values={fused_softmax_output[-1][-1][-1][:5].tolist()}, "
            f"\n > torch_values={torch_softmax_output[-1][-1][-1][:5].tolist()}"
        )


def test_layer_norm():
    bert = BertModel.from_pretrained("bert-base-cased").cuda().half()
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    test_text = (
        "Hello. How are you? I am fine thank you and you? yes Good. "
        "hi hi hi hi hi hi hi hi hi hi hi hi hi"  # 32
    )

    tokens = tokenizer(
        [test_text] * 4,
        return_tensors="pt",
    )

    # [bsz, seq_len, d_model]
    embedding_output = (
        bert.embeddings(
            input_ids=tokens["input_ids"].cuda(),
            position_ids=None,
            token_type_ids=tokens["token_type_ids"].cuda(),
            inputs_embeds=None,
            past_key_values_length=0,
        )
        .cuda()
        .half()
    )

    fused_layernorm_layer = (
        MixedFusedLayerNorm(normalized_shape=embedding_output.size(-1)).cuda().half()
    )

    torch_layernorm_layer = (
        LayerNorm(normalized_shape=embedding_output.size(-1)).cuda().half()
    )

    fused_output = fused_layernorm_layer(embedding_output)
    torch_output = torch_layernorm_layer(embedding_output)
    test_result = (fused_output - torch_output).abs()

    while test_result.dim() != 1:
        test_result = test_result.mean(dim=-1)

    diff = test_result.mean(dim=-1)

    if diff <= 1e-3:
        print(
            f"\n[Success] test_layer_norm"
            f"\n > mean_difference={diff}"
            f"\n > fused_values={fused_output[-1][-1][:5].tolist()}"
            f"\n > torch_values={torch_output[-1][-1][:5].tolist()}"
        )
    else:
        print(
            f"\n[Fail] test_layer_norm"
            f"\n > mean_difference={diff}, "
            f"\n > fused_values={fused_output[-1][-1][:5].tolist()}, "
            f"\n > torch_values={torch_output[-1][-1][:5].tolist()}"
        )


def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


def forward_torch_softmax(input, mask, scale):
    input = input * scale
    mask_output = attention_mask_func(input, mask) if mask is not None else input
    probs = torch.nn.Softmax(dim=-1)(mask_output)
    return probs


def test_masked_softmax_forward():
    import scaled_masked_softmax_cuda

    batch = 2
    attn = 16
    scale_t = torch.tensor([1.0])
    for qlen in [128, 256, 1024, 2048, 4096]:
        for klen in [128, 256, 1024, 2048]:
            inputs = torch.normal(0, 2, (batch, attn, qlen, klen), dtype=torch.float16, device='cuda:0')
            masks = torch.randint(0, 2, (batch, 1, qlen, klen), dtype=torch.bool, device='cuda:0')
            softmax_results = scaled_masked_softmax_cuda.forward(inputs, masks, scale_t[0].item())
            softmax_results_torch = forward_torch_softmax(inputs, masks, scale_t[0].item())
            error = (softmax_results_torch - softmax_results).abs().max()
            assert error < 1e-3

def test_masked_softmax_backward():
    import scaled_masked_softmax_cuda

    batch = 2
    attn = 16
    scale_t = torch.tensor([1.0])
    for qlen in [128, 256, 1024, 2048, 4096]:
        for klen in [128, 256, 1024, 2048]:
            inputs = torch.normal(0, 2, (batch, attn, qlen, klen), dtype=torch.float16, device='cuda:0')
            backward = torch.rand_like(inputs, dtype=torch.float16, device='cuda:0')
            masks = torch.randint(0, 2, (batch, 1, qlen, klen), dtype=torch.bool, device='cuda:0')
            softmax_results = scaled_masked_softmax_cuda.forward(inputs, masks, scale_t[0].item())
            back_grad = scaled_masked_softmax_cuda.backward(backward, softmax_results, scale_t[0].item())

            inputs.requires_grad = True
            softmax_results_torch = forward_torch_softmax(inputs, masks, scale_t[0].item())
            softmax_results_torch.backward(backward)
            error = (back_grad - inputs.grad).abs().max()
            assert error < 1e-3


def test_allmasked_softmax_forward():
    import scaled_masked_softmax_cuda

    batch = 2
    attn = 16
    scale_t = torch.tensor([1.0])
    for qlen in [128, 256, 1024, 2048, 4096]:
        for klen in [128, 256, 1024, 2048]:
            inputs = torch.normal(0, 2, (batch, attn, qlen, klen), dtype=torch.float16, device='cuda:0')
            masks = torch.ones((batch, 1, qlen, klen), dtype=torch.bool, device='cuda:0')
            softmax_results = scaled_masked_softmax_cuda.forward(inputs, masks, scale_t[0].item())
            softmax_results_torch = torch.zeros_like(inputs)
            error = (softmax_results_torch - softmax_results).abs().max()
            assert error == 0.0


def test_allmasked_softmax_backward():
    import scaled_masked_softmax_cuda

    batch = 2
    attn = 16
    scale_t = torch.tensor([1.0])
    for qlen in [128, 256, 1024, 2048, 4096]:
        for klen in [128, 256, 1024, 2048]:
            inputs = torch.normal(0, 2, (batch, attn, qlen, klen), dtype=torch.float16, device='cuda:0')
            backward = torch.rand_like(inputs, dtype=torch.float16, device='cuda:0')
            masks = torch.ones((batch, 1, qlen, klen), dtype=torch.bool, device='cuda:0')
            softmax_results = scaled_masked_softmax_cuda.forward(inputs, masks, scale_t[0].item())
            back_grad = scaled_masked_softmax_cuda.backward(backward, softmax_results, scale_t[0].item())
            inputs.requires_grad = True
            softmax_results_torch = forward_torch_softmax(inputs, masks, scale_t[0].item())
            softmax_results_torch.backward(backward)
            error = (back_grad - inputs.grad).abs().max()
            assert error < 1e-3

def test_fast_rotary_pos_emb_inner(dtype: torch.dtype, slice_input: bool, sq_axis: int):
    sq, b, np, hn = 2023, 6, 26, 1758
    from megatron.model.rotary_pos_embedding import RotaryEmbedding, apply_rotary_pos_emb
    device = "cuda"
    if slice_input:
        input_3x = torch.rand(sq, b, np, hn * 3, dtype=dtype, device=device)
        input = input_3x[..., :hn]
    else:
        input = torch.rand(sq, b, np, hn, dtype=dtype, device=device)
    input.requires_grad_()
    if sq_axis == 0:
        grad_output = torch.rand(input.shape, dtype=dtype, device=device)
    elif sq_axis == 1:
        grad_output_tr = torch.rand(input.shape[1], input.shape[0], input.shape[2], input.shape[3], dtype=dtype, device=device)
        grad_output = grad_output_tr.permute(1, 0, 2, 3)
    elif sq_axis == 2:
        grad_output_tr = torch.rand(input.shape[1], input.shape[2], input.shape[0], input.shape[3], dtype=dtype, device=device)
        grad_output = grad_output_tr.permute(2, 0, 1, 3)
    else:
        grad_output_tr = torch.rand(input.shape[1], input.shape[2], input.shape[3], input.shape[0], dtype=dtype, device=device)
        grad_output = grad_output_tr.permute(3, 0, 1, 2)

    rotary_pos_emb_ref = RotaryEmbedding(dim=hn).to(device).to(dtype)
    freqs_ref = rotary_pos_emb_ref(sq)
    output_ref = apply_rotary_pos_emb(input, freqs_ref)
    output_ref.backward(grad_output)
    d_input_ref = input.grad.clone()

    input.grad.zero_()

    rotary_pos_emb = RotaryEmbedding(dim=hn, use_fast_rope=True).to(device).to(dtype)
    freqs = rotary_pos_emb(sq)
    output = apply_rotary_pos_emb(input, freqs, use_fast_rope=True)
    output.backward(grad_output)
    d_input = input.grad.clone()

    print(f"fast_rotary_pos_emb {dtype} slice_input={slice_input} sq_axis={sq_axis} forward diff max", (output - output_ref).abs().max().tolist(), "mean", (output - output_ref).abs().mean().tolist())
    assert ((output - output_ref).abs() < 1e-4).all(), "forward error is too large"
    print(f"fast_rotary_pos_emb {dtype} slice_input={slice_input} sq_axis={sq_axis} backward diff max", (d_input - d_input_ref).abs().max().tolist(), "mean", (d_input - d_input_ref).abs().mean().tolist())
    assert ((d_input - d_input_ref).abs() < 1e-4).all(), "backward error is too large"


def test_fast_rotary_pos_emb():
    test_fast_rotary_pos_emb_inner(torch.bfloat16, False, 0)
    test_fast_rotary_pos_emb_inner(torch.bfloat16, True, 1)
    test_fast_rotary_pos_emb_inner(torch.bfloat16, True, 2)
    test_fast_rotary_pos_emb_inner(torch.bfloat16, True, 3)
    test_fast_rotary_pos_emb_inner(torch.float16, False, 0)
    test_fast_rotary_pos_emb_inner(torch.float16, True, 1)
    test_fast_rotary_pos_emb_inner(torch.float16, True, 2)
    test_fast_rotary_pos_emb_inner(torch.float16, True, 3)
    test_fast_rotary_pos_emb_inner(torch.float32, False, 0)
    test_fast_rotary_pos_emb_inner(torch.float32, True, 1)
    test_fast_rotary_pos_emb_inner(torch.float32, True, 2)
    test_fast_rotary_pos_emb_inner(torch.float32, True, 3)


def test_flip():
    from megatron.core.context_parallel.dattention import flip_cp_, flip_cp, slice_cp
    contiguous_tensor = torch.tensor([
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11,
        12, 13, 14, 15,
    ], device="cuda")
    interleaved_tensor = torch.tensor([
        0, 1, 14, 15,
        4, 5, 10, 11,
        8, 9, 6, 7,
        12, 13, 2, 3,
    ], device="cuda")
    slice_tensor_rank1 = torch.tensor([
        4, 5, 10, 11,
    ], device="cuda")
    x = contiguous_tensor.clone()
    flip_cp_(x, dim=0, world_size=4)
    assert (x == interleaved_tensor).all()
    flip_cp_(x, dim=0, world_size=4)
    assert (x == contiguous_tensor).all()
    assert (flip_cp(contiguous_tensor, dim=0, world_size=4) == interleaved_tensor).all()
    assert (flip_cp(interleaved_tensor, dim=0, world_size=4) == contiguous_tensor).all()
    assert (slice_cp(contiguous_tensor, dim=0, world_size=4, rank=1) == slice_tensor_rank1).all()


def test_addmm_inplace():
    from megatron.fused_kernels.wrap_gemm_api import addmm_inplace
    m, n, k = 2048, 3072, 3072
    input = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
    mat1 = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    mat2 = torch.randn(k, n, dtype=torch.bfloat16, device="cuda") / k ** .5
    answer = torch.addmm(input, mat1, mat2)
    output = addmm_inplace(input.clone(), mat1, mat2)
    print(f"addmm_inplace diff max {(output - answer).abs().max()} diff mean {(output - answer).abs().mean()} answer max {answer.abs().max()}")
    assert (output - answer).abs().max().item() < .01


def test_fast_cat():
    import fast_cat_cuda
    x1 = torch.rand(131072, 3072, dtype=torch.bfloat16, device="cuda")
    x2 = torch.rand(131072, 3072, dtype=torch.bfloat16, device="cuda")
    x3 = torch.rand(131072, 3072, dtype=torch.bfloat16, device="cuda")
    output = torch.empty(x1.shape[0], 3 * x1.shape[1], dtype=torch.bfloat16, device="cuda")
    for _ in range(10):
        fast_cat_cuda.cat([x1.data_ptr(), x2.data_ptr(), x3.data_ptr()], output.data_ptr(),
                          x1.shape[0], x1.shape[1] * x1.element_size(), torch.cuda.current_stream().cuda_stream)
        answer = torch.cat([x1, x2, x3], dim=1)
        answer.clone()
    assert (output == answer).all()


if __name__ == "__main__":
    try:
        from transformers import BertTokenizer, GPT2Tokenizer
        from transformers.models.bert.modeling_bert import BertModel
        from transformers.models.gpt2.modeling_gpt2 import GPT2Model
        import transformers

        transformers.logging.set_verbosity(
            transformers.logging.FATAL,
        )

    except:
        print("\n[Fail] Please install `transformers` package to test fused kernels\n")
        exit(-1)

    class Args:
        rank = 0
        masked_softmax_fusion = True
        use_fast_rope = True
        context_parallel_size = 2
    load(Args())
    test_masked_softmax_forward()
    test_masked_softmax_backward()
    test_allmasked_softmax_forward()
    test_allmasked_softmax_backward()
    test_load_fused_kernels()
    test_fused_softmax()
    test_fused_upper_triangle_mask_softmax()
    test_layer_norm()
    test_fast_rotary_pos_emb()
    test_flip()
    test_addmm_inplace()
    test_fast_cat()
