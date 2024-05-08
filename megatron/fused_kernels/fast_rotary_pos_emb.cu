// Copyright (c) 2024, Kuaishou Technology. All rights reserved.

#include "fast_rotary_pos_emb.h"

#include <iostream>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/core/ScalarType.h>

namespace {

template<typename IntType>
constexpr IntType ceildiv(IntType a, IntType b) {
    return (a + b - 1) / b;
}

template<typename T> __device__ inline T math_sin(T x) { return hsin(x); }
template<typename T> __device__ inline T math_cos(T x) { return hcos(x); }
template<> __device__ inline float math_sin<float>(float x) { return sin(x); }
template<> __device__ inline float math_cos<float>(float x) { return cos(x); }
#if __CUDA_ARCH__ < 800
template<> __device__ inline __nv_bfloat16 math_sin<__nv_bfloat16>(__nv_bfloat16 x) { return __float2bfloat16(sin(__bfloat162float(x))); }
template<> __device__ inline __nv_bfloat16 math_cos<__nv_bfloat16>(__nv_bfloat16 x) { return __float2bfloat16(cos(__bfloat162float(x))); }
#endif

template<bool precompute_sin_cos, typename T>
__device__ inline T get_sin(T const *freqs, int idx) {
    if (precompute_sin_cos) {
        return freqs[idx * 2];
    } else {
        return math_sin(freqs[idx]);
    }
}

template<bool precompute_sin_cos, typename T>
__device__ inline T get_cos(T const *freqs, int idx) {
    if (precompute_sin_cos) {
        return freqs[idx * 2 + 1];
    } else {
        return math_cos(freqs[idx]);
    }
}

}  // namespace

template<bool precompute_sin_cos, typename T>
__global__ void fast_rotary_pos_emb_forward_kernel(T const *t, T const *freqs, T *output, int sq, int b, int np, int hn, int stride_sq, int stride_np) {
    int i_hn = threadIdx.x;
    int i_sq = blockIdx.x;
    int i_b = blockIdx.y;
    for (int i_np = threadIdx.y; i_np < np; i_np += blockDim.y) {
        int64_t idx_output = ((i_b * sq + i_sq) * np + i_np) * (int64_t)hn + i_hn;
        int64_t idx_t = i_sq * (int64_t)stride_sq + (i_b * np + i_np) * (int64_t)stride_np + i_hn;
        int idx_freqs = i_sq * hn + i_hn;
        output[idx_output] = t[idx_t] * get_cos<precompute_sin_cos>(freqs, idx_freqs) + -t[idx_t + hn / 2] * get_sin<precompute_sin_cos>(freqs, idx_freqs);
        output[idx_output + hn / 2] = t[idx_t + hn / 2] * get_cos<precompute_sin_cos>(freqs, idx_freqs + hn / 2) + t[idx_t] * get_sin<precompute_sin_cos>(freqs, idx_freqs + hn / 2);
    }
}

template<bool precompute_sin_cos, typename T>
__global__ void fast_rotary_pos_emb_backward_kernel(T const *grad_output, T const *freqs, T *d_t, int sq, int b, int np, int hn, int stride_sq, int stride_b, int stride_np, int stride_hn) {
    int i_hn = threadIdx.x;
    int i_sq = blockIdx.x;
    int i_b = blockIdx.y;
    for (int i_np = threadIdx.y; i_np < np; i_np += blockDim.y) {
        int64_t idx_t = ((i_sq * b + i_b) * np + i_np) * (int64_t)hn + i_hn;
        int64_t idx_o = i_sq * (int64_t)stride_sq + i_b * (int64_t)stride_b + i_np * (int64_t)stride_np + i_hn * (int64_t)stride_hn;
        int idx_freqs = i_sq * hn + i_hn;
        d_t[idx_t] = grad_output[idx_o] * get_cos<precompute_sin_cos>(freqs, idx_freqs) + grad_output[idx_o + hn / 2 * stride_hn] * get_sin<precompute_sin_cos>(freqs, idx_freqs + hn / 2);
        d_t[idx_t + hn / 2] = grad_output[idx_o + hn / 2 * stride_hn] * get_cos<precompute_sin_cos>(freqs, idx_freqs + hn / 2) - grad_output[idx_o] * get_sin<precompute_sin_cos>(freqs, idx_freqs);
    }
}

template<typename T>
int fast_rotary_pos_emb_forward_op_dtype(void const *t, void const *freqs, void *output, int sq, int b, int np, int hn, int stride_sq, int stride_np, bool precompute_sin_cos, cudaStream_t stream) {
    T const *t_T = reinterpret_cast<T const *>(t);
    T const *freqs_T = reinterpret_cast<T const *>(freqs);
    T *output_T = reinterpret_cast<T *>(output);
    dim3 grid(sq, b);
    dim3 block(hn / 2, ceildiv(256, hn / 2));
    if (precompute_sin_cos)
        fast_rotary_pos_emb_forward_kernel<true><<<grid, block, 0, stream>>>(t_T, freqs_T, output_T, sq, b, np, hn, stride_sq, stride_np);
    else
        fast_rotary_pos_emb_forward_kernel<false><<<grid, block, 0, stream>>>(t_T, freqs_T, output_T, sq, b, np, hn, stride_sq, stride_np);
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error " << (int)err << ": " << cudaGetErrorString(err) << " " << __FILE__ << ":" << __LINE__ << std::endl;
        return 1;
    }
    return 0;
}

int fast_rotary_pos_emb_forward_op(void const *t, void const *freqs, void *output, int sq, int b, int np, int hn, int stride_sq, int stride_np, int dtype, bool precompute_sin_cos, cudaStream_t stream) {
    switch ((c10::ScalarType)dtype) {
    case c10::ScalarType::BFloat16:
        return fast_rotary_pos_emb_forward_op_dtype<__nv_bfloat16>(t, freqs, output, sq, b, np, hn, stride_sq, stride_np, precompute_sin_cos, stream);
    case c10::ScalarType::Half:
        return fast_rotary_pos_emb_forward_op_dtype<__half>(t, freqs, output, sq, b, np, hn, stride_sq, stride_np, precompute_sin_cos, stream);
    case c10::ScalarType::Float:
        return fast_rotary_pos_emb_forward_op_dtype<float>(t, freqs, output, sq, b, np, hn, stride_sq, stride_np, precompute_sin_cos, stream);
    default:
        std::cerr << "Unsupported dtype " << c10::toString((c10::ScalarType)dtype) << " " << __FILE__ << ":" << __LINE__ << std::endl;
        return 1;
    }
}

template<typename T>
int fast_rotary_pos_emb_backward_op_dtype(void const *grad_output, void const *freqs, void *d_t, int sq, int b, int np, int hn, int stride_sq, int stride_b, int stride_np, int stride_hn, bool precompute_sin_cos, cudaStream_t stream) {
    T const *grad_output_T = reinterpret_cast<T const *>(grad_output);
    T const *freqs_T = reinterpret_cast<T const *>(freqs);
    T *d_t_T = reinterpret_cast<T *>(d_t);
    dim3 grid(sq, b);
    dim3 block(hn / 2, ceildiv(256, hn / 2));
    if (precompute_sin_cos)
        fast_rotary_pos_emb_backward_kernel<true><<<grid, block, 0, stream>>>(grad_output_T, freqs_T, d_t_T, sq, b, np, hn, stride_sq, stride_b, stride_np, stride_hn);
    else
        fast_rotary_pos_emb_backward_kernel<false><<<grid, block, 0, stream>>>(grad_output_T, freqs_T, d_t_T, sq, b, np, hn, stride_sq, stride_b, stride_np, stride_hn);
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error " << (int)err << ": " << cudaGetErrorString(err) << " " << __FILE__ << ":" << __LINE__ << std::endl;
        return 1;
    }
    return 0;
}

int fast_rotary_pos_emb_backward_op(void const *grad_output, void const *freqs, void *d_t, int sq, int b, int np, int hn, int stride_sq, int stride_b, int stride_np, int stride_hn, int dtype, bool precompute_sin_cos, cudaStream_t stream) {
    switch ((c10::ScalarType)dtype) {
    case c10::ScalarType::BFloat16:
        return fast_rotary_pos_emb_backward_op_dtype<__nv_bfloat16>(grad_output, freqs, d_t, sq, b, np, hn, stride_sq, stride_b, stride_np, stride_hn, precompute_sin_cos, stream);
    case c10::ScalarType::Half:
        return fast_rotary_pos_emb_backward_op_dtype<__half>(grad_output, freqs, d_t, sq, b, np, hn, stride_sq, stride_b, stride_np, stride_hn, precompute_sin_cos, stream);
    case c10::ScalarType::Float:
        return fast_rotary_pos_emb_backward_op_dtype<float>(grad_output, freqs, d_t, sq, b, np, hn, stride_sq, stride_b, stride_np, stride_hn, precompute_sin_cos, stream);
    default:
        std::cerr << "Unsupported dtype " << c10::toString((c10::ScalarType)dtype) << " " << __FILE__ << ":" << __LINE__ << std::endl;
        return 1;
    }
}
