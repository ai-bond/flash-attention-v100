// ============================================================================
// * Copyright (c) 2025, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ============================================================================
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <stdexcept>
#include "fused_mha.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashAttention-2 implementation optimized for Volta";
    m.def("fwd", &flash_attention_forward, "FlashAttention-2 Forward Pass (Volta)");
    m.def("bwd", &flash_attention_backward, "FlashAttention-2 Backward Pass (Volta)");
}

