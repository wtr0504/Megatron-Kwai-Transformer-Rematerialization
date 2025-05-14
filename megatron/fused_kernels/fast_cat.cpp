// Copyright (c) 2024, Kuaishou Technology. All rights reserved.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "fast_cat_cuda.hpp"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cat", &fast_cat::cat_cuda);
}
