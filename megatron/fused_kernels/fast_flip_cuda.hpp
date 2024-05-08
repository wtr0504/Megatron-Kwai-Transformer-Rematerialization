// Copyright (c) 2024, Kuaishou Technology. All rights reserved.

#pragma once

#include <cstddef>
#include <cstdint>


namespace fast_flip {

void flip(intptr_t output_int, intptr_t input_int, size_t batch_size, size_t batch_stride, size_t axis_size, size_t axis_stride, size_t inner_size, size_t element_size, intptr_t stream_int);

}
