// Copyright (c) 2024, Kuaishou Technology. All rights reserved.

#pragma once

#include <cstdint>


namespace wrap_gemm {

void wrap_gemm_bf16bf16bf16_f32_nn_beta1_cuda(intptr_t A_intptr, intptr_t B_intptr, intptr_t C_intptr, int m, int n, int k, intptr_t handle_intptr);

void wrap_cuda_memcpy_2d_async(intptr_t dst_intptr, size_t dpitch, intptr_t src_intptr, size_t spitch, size_t width, size_t height, int cuda_memcpy_kind, intptr_t stream_intptr);

}
