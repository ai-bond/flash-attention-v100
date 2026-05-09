// ============================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ============================================================================
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <stdexcept>
#include "mha.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashAttention implementation optimized for Volta";
    m.def("fwd", &flash_attention_forward,  "FlashAttention Forward Pass");
    m.def("bwd", &flash_attention_backward, "FlashAttention Backward Pass");
    m.def("varlen_fwd", &flash_attention_varlen_forward,  "FlashAttention Forward Pass  (variable length)");
}
