// Copyright (c) 2024, Kuaishou Technology. All rights reserved.

#include <cstdio>
#include <stdexcept>

#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include "fast_rotary_pos_emb.h"

torch::Tensor fast_rotary_pos_emb_forward(torch::Tensor t, torch::Tensor freqs, bool precompute_sin_cos) {
    if (t.dtype() != freqs.dtype() ||
            (t.dtype() != torch::kBFloat16 && t.dtype() != torch::kFloat16 && t.dtype() != torch::kFloat32))
        throw std::invalid_argument("fast_rotary_pos_emb_forward input dtype error, only BFloat16, Float16, and Float32 are supported");
    if (t.dim() != 4 || freqs.dim() != 4 ||
            t.size(0) != freqs.size(0) ||
            freqs.size(1) != 1 ||
            freqs.size(2) != 1 ||
            t.size(3) * (1 + (int)precompute_sin_cos) != freqs.size(3) ||
            t.size(3) % 2 != 0)
        throw std::invalid_argument("fast_rotary_pos_emb_forward input shape error");
    if (!t.is_contiguous()) {
        if (t.stride(3) != 1 ||
                t.stride(1) != t.size(2) * t.stride(2))
            throw std::invalid_argument("fast_rotary_pos_emb_forward input 0 is neither contiguous nor sliced at dim=2");
    }
    if (!freqs.is_contiguous()) {
        throw std::invalid_argument("fast_rotary_pos_emb_forward input is not contiguous");
    }
    auto options = torch::TensorOptions()
        .dtype(t.dtype())
        .device(t.device());
    auto output = torch::empty_strided(t.sizes(), {t.size(3) * t.size(2), t.size(3) * t.size(2) * t.size(0), t.size(3), 1}, options);
    int ret = fast_rotary_pos_emb_forward_op(t.data_ptr(), freqs.data_ptr(), output.data_ptr(), t.size(0), t.size(1), t.size(2), t.size(3), t.stride(0), t.stride(2), (int)t.dtype().toScalarType(), precompute_sin_cos, c10::cuda::getCurrentCUDAStream().stream());
    if (ret != 0)
        throw std::runtime_error("fast_rotary_pos_emb_forward failed");
    return output;
}

torch::Tensor fast_rotary_pos_emb_backward(torch::Tensor grad_output, torch::Tensor freqs, bool precompute_sin_cos) {
    if (grad_output.dtype() != freqs.dtype() ||
            (grad_output.dtype() != torch::kBFloat16 && grad_output.dtype() != torch::kFloat16 && grad_output.dtype() != torch::kFloat32))
        throw std::invalid_argument("fast_rotary_pos_emb_backward input dtype error, only BFloat16, Float16, and Float32 are supported");
    if (grad_output.dim() != 4 || freqs.dim() != 4 ||
            grad_output.size(0) != freqs.size(0) ||
            freqs.size(1) != 1 ||
            freqs.size(2) != 1 ||
            grad_output.size(3) * (1 + (int)precompute_sin_cos) != freqs.size(3) ||
            grad_output.size(3) % 2 != 0)
        throw std::invalid_argument("fast_rotary_pos_emb_backward input shape error");
    if (!freqs.is_contiguous()) {
        throw std::invalid_argument("fast_rotary_pos_emb_backward input is not contiguous");
    }
    auto options = torch::TensorOptions()
        .dtype(grad_output.dtype())
        .device(grad_output.device());
    auto d_t = torch::empty(grad_output.sizes(), options);
    int ret = fast_rotary_pos_emb_backward_op(grad_output.data_ptr(), freqs.data_ptr(), d_t.data_ptr(), grad_output.size(0), grad_output.size(1), grad_output.size(2), grad_output.size(3), grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3), (int)grad_output.dtype().toScalarType(), precompute_sin_cos, c10::cuda::getCurrentCUDAStream().stream());
    if (ret != 0)
        throw std::runtime_error("fast_rotary_pos_emb_backward failed");
    return d_t;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fast_rotary_pos_emb_forward);
    m.def("backward", &fast_rotary_pos_emb_backward);
}
