// Copyright (c) 2024, Kuaishou Technology. All rights reserved.

#include "wrap_gemm_cuda.hpp"

#include <iostream>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cublas_v2.h>


namespace wrap_gemm {

// C += A * B
void wrap_gemm_bf16bf16bf16_f32_nn_beta1_cuda(intptr_t A_intptr, intptr_t B_intptr, intptr_t C_intptr, int m, int n, int k, intptr_t handle_intptr) {
    __nv_bfloat16 const *A = reinterpret_cast<__nv_bfloat16 const *>(A_intptr);
    __nv_bfloat16 const *B = reinterpret_cast<__nv_bfloat16 const *>(B_intptr);
    __nv_bfloat16 *C = reinterpret_cast<__nv_bfloat16 *>(C_intptr);
    cublasHandle_t handle = reinterpret_cast<cublasHandle_t>(handle_intptr);

    float alpha = 1.;
    float beta = 1.;

    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        B,
        CUDA_R_16BF,
        n,
        A,
        CUDA_R_16BF,
        k,
        &beta,
        C,
        CUDA_R_16BF,
        n,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS error " << (int)status << ": " << cublasGetStatusString(status) << " " << __FILE__ << ":" << __LINE__ << std::endl;
        throw std::runtime_error("cuBLAS error " + std::to_string(status) + ": " + cublasGetStatusString(status));
    }
}

void wrap_cuda_memcpy_2d_async(intptr_t dst_intptr, size_t dpitch, intptr_t src_intptr, size_t spitch, size_t width, size_t height, int cuda_memcpy_kind, intptr_t stream_intptr) {
    void *dst = reinterpret_cast<void *>(dst_intptr);
    void const *src = reinterpret_cast<void const *>(src_intptr);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_intptr);
    cudaError_t err = cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, (cudaMemcpyKind)cuda_memcpy_kind, stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error " << (int)err << ": " << cudaGetErrorString(err) << " " << __FILE__ << ":" << __LINE__ << std::endl;
        throw std::runtime_error("CUDA error " + std::to_string(err) + ": " + cudaGetErrorString(err));
    }
}

}
