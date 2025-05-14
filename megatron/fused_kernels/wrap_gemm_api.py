# Copyright (c) 2024, Kuaishou Technology. All rights reserved.

import torch
import wrap_gemm_cuda


def addmm_inplace(input, mat1, mat2):
    assert input.is_cuda and mat1.is_cuda and mat2.is_cuda
    assert input.is_contiguous() and mat1.is_contiguous() and mat2.is_contiguous()
    assert mat2.dim() == 2 and mat1.shape[-1] == mat2.shape[0]
    assert input.shape[:-1] == mat1.shape[:-1] and input.shape[-1] == mat2.shape[-1]
    if input.dtype == torch.bfloat16 and mat1.dtype == torch.bfloat16 and mat2.dtype == torch.bfloat16:
        wrap_gemm_cuda.wrap_gemm_bf16bf16bf16_f32_nn_beta1_cuda(
            mat1.data_ptr(),
            mat2.data_ptr(),
            input.data_ptr(),
            mat1.shape[:-1].numel(),
            mat2.shape[-1],
            mat2.shape[0],
            torch.cuda.current_blas_handle(),
        )
    else:
        raise NotImplementedError(f"not implemented addmm_inplace with {input.dtype} {mat1.dtype} {mat2.dtype}")
    return input
