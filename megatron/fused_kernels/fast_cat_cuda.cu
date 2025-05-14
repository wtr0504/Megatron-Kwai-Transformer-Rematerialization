// Copyright (c) 2024, Kuaishou Technology. All rights reserved.

#include "fast_cat_cuda.hpp"

#include <iostream>
#include <stdexcept>


namespace fast_cat {

constexpr static const int MAX_NUM_INPUTS = 4;
constexpr static const int BLOCK_DIM = 256;
constexpr static const int MAX_GRID_DIM_Y = 32768;

template<typename T>
struct Params {
    T const *inputs[MAX_NUM_INPUTS];
    T *output;
    size_t inner_numel;
    size_t outer_size;
};

template<typename IntType>
constexpr IntType ceildiv(IntType a, IntType b) {
    return (a + b - 1) / b;
}

template<typename T, int NUM_INPUTS>
__global__ void cat_cuda_kernel(Params<T> params) {
    int outer_idx = blockIdx.y + blockIdx.z * MAX_GRID_DIM_Y;
    int inner_idx = blockIdx.x * BLOCK_DIM + threadIdx.x;
    if (outer_idx < params.outer_size) {
        if (inner_idx < params.inner_numel) {
#pragma unroll
            for (int input_idx = 0; input_idx < NUM_INPUTS; input_idx++) {
                params.output[(outer_idx * NUM_INPUTS + input_idx) * params.inner_numel + inner_idx] =
                    params.inputs[input_idx][outer_idx * params.inner_numel + inner_idx];
            }
        }
    }
}

template<typename T, int NUM_INPUTS>
auto dispatch_cat_cuda_kernel(int num_inputs) {
    if (num_inputs == NUM_INPUTS)
        return &cat_cuda_kernel<T, NUM_INPUTS>;
    if constexpr (NUM_INPUTS >= 1)
        return dispatch_cat_cuda_kernel<T, NUM_INPUTS - 1>(num_inputs);
    else
        throw std::invalid_argument("num_inputs does not match any template");
}

void cat_cuda(std::vector<intptr_t> const &inputs_intptr, intptr_t output_intptr, size_t outer_size, size_t inner_size, intptr_t stream_intptr) {
    using T = uint64_t;
    Params<T> params;
    if (inputs_intptr.size() > MAX_NUM_INPUTS)
        throw std::invalid_argument("inputs size (" + std::to_string(inputs_intptr.size()) + ") exceeds MAX_NUM_INPUTS (" + std::to_string(MAX_NUM_INPUTS) + ")");
    if (inner_size % sizeof(T) != 0)
        throw std::invalid_argument("inner size (" + std::to_string(inner_size) + ") is not divisible by " + std::to_string(sizeof(T)));
    for (size_t i = 0; i < inputs_intptr.size(); i++) {
        params.inputs[i] = reinterpret_cast<T const *>(inputs_intptr[i]);
    }
    params.output = reinterpret_cast<T *>(output_intptr);
    params.inner_numel = inner_size / sizeof(T);
    params.outer_size = outer_size;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_intptr);
    dim3 grid(ceildiv(params.inner_numel, (size_t)BLOCK_DIM), std::min(outer_size, (size_t)MAX_GRID_DIM_Y), ceildiv(outer_size, (size_t)MAX_GRID_DIM_Y));
    dispatch_cat_cuda_kernel<T, MAX_NUM_INPUTS>(inputs_intptr.size())<<<grid, BLOCK_DIM, 0, stream>>>(params);
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error " << (int)err << ": " << cudaGetErrorString(err) << " " << __FILE__ << ":" << __LINE__ << std::endl;
        throw std::runtime_error("CUDA error " + std::to_string(err) + ": " + cudaGetErrorString(err));
    }
}

}
