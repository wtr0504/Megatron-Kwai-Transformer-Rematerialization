// Copyright (c) 2024, Kuaishou Technology. All rights reserved.

#pragma once

#include <cstdint>
#include <vector>


namespace fast_cat {

void cat_cuda(std::vector<intptr_t> const &inputs_intptr, intptr_t output_intptr, size_t outer_size, size_t inner_size, intptr_t stream_intptr);

}
