// Copyright (c) 2024, Kuaishou Technology. All rights reserved.

#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

int fast_rotary_pos_emb_forward_op(void const *t, void const *freqs, void *output, int sq, int b, int np, int hn, int stride_sq, int stride_np, int dtype, bool precompute_sin_cos, cudaStream_t stream);

int fast_rotary_pos_emb_backward_op(void const *grad_output, void const *freqs, void *d_t, int sq, int b, int np, int hn, int stride_sq, int stride_b, int stride_np, int stride_hn, int dtype, bool precompute_sin_cos, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
