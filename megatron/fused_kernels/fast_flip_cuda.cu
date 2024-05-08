// Copyright (c) 2024, Kuaishou Technology. All rights reserved.

#include "fast_flip_cuda.hpp"

#include <iostream>
#include <stdexcept>


namespace fast_flip {

template<typename IntType>
constexpr IntType ceildiv(IntType a, IntType b) {
    return (a + b - 1) / b;
}

template<typename T>
__global__ void flip_kernel(T *output, T const *input, size_t batch_stride, size_t axis_size, size_t axis_stride, size_t inner_elements) {
    int inner_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (inner_idx < inner_elements) {
        int idx_a = blockIdx.z * batch_stride + blockIdx.y * axis_stride + inner_idx;
        int idx_b = blockIdx.z * batch_stride + (axis_size - blockIdx.y - 1) * axis_stride + inner_idx;
        T a = input[idx_a];
        output[idx_a] = input[idx_b];
        output[idx_b] = a;
    }
}

void flip(intptr_t output_int, intptr_t input_int, size_t batch_size, size_t batch_stride, size_t axis_size, size_t axis_stride, size_t inner_size, size_t element_size, intptr_t stream_int) {
    using T = uint64_t;
    T *output = reinterpret_cast<T *>(output_int);
    T const *input = reinterpret_cast<T const *>(input_int);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_int);
    size_t inner_bytes = inner_size * element_size;
    if (axis_size <= 1)
        return;
    if (inner_bytes % sizeof(T) != 0)
        throw std::runtime_error("(inner_size * element_size) % " + std::to_string(sizeof(T)) + " != 0");
    if (output_int % sizeof(T) != 0 || input_int % sizeof(T) != 0)
        throw std::runtime_error("output/input ptr is not aligned");
    if (batch_stride * element_size % sizeof(T) != 0 || axis_stride * element_size % sizeof(T) != 0)
        throw std::runtime_error("batch/axis stride is not aligned");
    dim3 grid(ceildiv(inner_bytes / sizeof(T), size_t(256)), axis_size / 2, batch_size);
    flip_kernel<<<grid, 256, 0, stream>>>(output, input, batch_stride * element_size / sizeof(T), axis_size, axis_stride * element_size / sizeof(T), inner_bytes / sizeof(T));
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error " << (int)err << ": " << cudaGetErrorString(err) << " " << __FILE__ << ":" << __LINE__ << std::endl;
        throw std::runtime_error("CUDA error " + std::to_string(err) + ": " + cudaGetErrorString(err));
    }
}

}
