// Copyright (c) 2024, Kuaishou Technology. All rights reserved.

#include <pybind11/pybind11.h>

#include "fast_flip_cuda.hpp"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flip", &fast_flip::flip);
}
