// Copyright (c) 2024, Kuaishou Technology. All rights reserved.

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include "wrap_gemm_cuda.hpp"


namespace wrap_gemm {

torch::Tensor cuda_malloc_host(size_t size) {
    void *p;
    cudaError_t err = cudaMallocHost(&p, size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error " << (int)err << ": " << cudaGetErrorString(err) << " " << __FILE__ << ":" << __LINE__ << std::endl;
        abort();
    }
    auto deleter = [](void *p) {
        cudaError_t err = cudaFreeHost(p);
        if (err != cudaSuccess) {
            std::cerr << "CUDA error " << (int)err << ": " << cudaGetErrorString(err) << " " << __FILE__ << ":" << __LINE__ << std::endl;
            abort();
        }
    };
    auto options = torch::TensorOptions().dtype(torch::kUInt8);
    return torch::from_blob(p, {(int64_t)size}, {1}, deleter, options);
}

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wrap_gemm_bf16bf16bf16_f32_nn_beta1_cuda", &wrap_gemm::wrap_gemm_bf16bf16bf16_f32_nn_beta1_cuda);
    m.def("wrap_cuda_memcpy_2d_async", &wrap_gemm::wrap_cuda_memcpy_2d_async);
    m.def("wrap_cuda_malloc_host", &wrap_gemm::cuda_malloc_host);
}
