// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <stdexcept>
#include <optional>
#include "mha.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// ======================================================================================
// PyBind11
// ======================================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashAttention implementation optimized for Volta";
    m.def("fwd",          &flash_attention_forward,        "FlashAttention Forward Pass");
    m.def("bwd",          &flash_attention_backward,       "FlashAttention Backward Pass");
    m.def("varlen_fwd",   &flash_attention_varlen_forward, "FlashAttention Forward Pass (variable length)");
    m.def("varlen_bwd",   &flash_attention_varlen_backward,"FlashAttention Backward Pass (variable length)");
    m.def("fwd_kvcache",  &flash_attention_kvcache,        "Forward pass with KV-cache");
}

PYBIND11_MODULE(flash_attn_2_cuda, n) {
    n.doc() = "FA2 compatibility wrapper";
    n.def("fwd",          &flash_attention_forward,        "FlashAttention Forward Pass");
    n.def("bwd",          &flash_attention_backward,       "FlashAttention Backward Pass");
    n.def("varlen_fwd",   &flash_attention_varlen_forward, "FlashAttention Forward Pass (variable length)");
    n.def("varlen_bwd",   &flash_attention_varlen_backward,"FlashAttention Backward Pass (variable length)");
    n.def("fwd_kvcache",  &flash_attention_kvcache,        "Forward pass with KV-cache");
}

// ======================================================================================
// TorchBind
// ======================================================================================
inline std::optional<const at::Tensor> convert_opt(const std::optional<at::Tensor>& opt) {
    if (opt.has_value()) {
        return std::optional<const at::Tensor>(std::in_place, opt.value());
    }
    return std::nullopt;
}

// fwd
std::vector<at::Tensor> torchbind_fwd(
          at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    std::optional<at::Tensor> out,
    std::optional<at::Tensor> alibi_slopes,
    double  p_dropout,
    double  softmax_scale,
    bool    is_causal,
    int64_t window_left,
    int64_t window_right,
    double  softcap,
    bool    return_softmax,
    std::optional<at::Generator> gen
) {
    std::optional<at::Tensor> out_ref   = out;
    std::optional<at::Tensor> alibi_ref = alibi_slopes;

    return flash_attention_forward(
        q,
        k,
        v,
        out_ref,
        alibi_ref,
        static_cast<float>(p_dropout),
        static_cast<float>(softmax_scale),
        is_causal,
        static_cast<int>(window_left),
        static_cast<int>(window_right),
        static_cast<float>(softcap),
        return_softmax,
        gen
    );
}

// bwd
std::vector<at::Tensor> torchbind_bwd(
    const at::Tensor& dout,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& out,
    const at::Tensor& softmax_lse,
    std::optional<at::Tensor> dq,
    std::optional<at::Tensor> dk,
    std::optional<at::Tensor> dv,
    std::optional<at::Tensor> alibi_slopes,
    double  p_dropout,
    double  softmax_scale,
    bool    is_causal,
    int64_t window_left,
    int64_t window_right,
    double  softcap,
    bool    deterministic,
    std::optional<at::Generator> gen,
    std::optional<at::Tensor> rng_state
) {
    std::optional<at::Tensor> dq_ref    = dq;
    std::optional<at::Tensor> dk_ref    = dk;
    std::optional<at::Tensor> dv_ref    = dv;
    std::optional<at::Tensor> alibi_ref = alibi_slopes;
    std::optional<at::Tensor> rng_ref   = rng_state;

    return flash_attention_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq_ref,
        dk_ref,
        dv_ref,
        alibi_ref,
        static_cast<float>(p_dropout),
        static_cast<float>(softmax_scale),
        is_causal,
        static_cast<int>(window_left),
        static_cast<int>(window_right),
        static_cast<float>(softcap),
        deterministic,
        gen,
        rng_ref
    );
}

// varlen_fwd
std::vector<at::Tensor> torchbind_varlen_fwd(
          at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    std::optional<at::Tensor> out,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    std::optional<at::Tensor> seqused_k,
    std::optional<at::Tensor> leftpad_k,
    std::optional<at::Tensor> block_table,
    std::optional<at::Tensor> alibi_slopes,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    double  p_dropout,
    double  softmax_scale,
    bool    zero_tensors,
    bool    is_causal,
    int64_t window_left,
    int64_t window_right,
    double  softcap,
    bool    return_softmax,
    std::optional<at::Generator> gen,
    int64_t num_splits
) {
    std::optional<at::Tensor> out_ref           = out;
    std::optional<at::Tensor> seqused_ref       = seqused_k;
    std::optional<const at::Tensor> leftpad_ref = convert_opt(leftpad_k);
    std::optional<at::Tensor> block_ref         = block_table;
    std::optional<at::Tensor> alibi_ref         = alibi_slopes;

    return flash_attention_varlen_forward(
        q,
        k,
        v,
        out_ref,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_ref,
        leftpad_ref,
        block_ref,
        alibi_ref,
        static_cast<int>(max_seqlen_q),
        static_cast<int>(max_seqlen_k),
        static_cast<float>(p_dropout),
        static_cast<float>(softmax_scale),
        zero_tensors,
        is_causal,
        static_cast<int>(window_left),
        static_cast<int>(window_right),
        static_cast<float>(softcap),
        return_softmax,
        gen,
        static_cast<int>(num_splits)
    );
}

// varlen_bwd
std::vector<at::Tensor> torchbind_varlen_bwd(
    const at::Tensor& dout,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& out,
    const at::Tensor& softmax_lse,
    std::optional<at::Tensor> dq,
    std::optional<at::Tensor> dk,
    std::optional<at::Tensor> dv,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    std::optional<at::Tensor> alibi_slopes,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    double  p_dropout,
    double  softmax_scale,
    bool    zero_tensors,
    bool    is_causal,
    int64_t window_left,
    int64_t window_right,
    double  softcap,
    bool deterministic,
    std::optional<at::Generator> gen,
    std::optional<at::Tensor> rng_state
) {
    std::optional<at::Tensor> dq_ref    = dq;
    std::optional<at::Tensor> dk_ref    = dk;
    std::optional<at::Tensor> dv_ref    = dv;
    std::optional<at::Tensor> alibi_ref = alibi_slopes;
    std::optional<at::Tensor> rng_ref   = rng_state;

    return flash_attention_varlen_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq_ref,
        dk_ref,
        dv_ref,
        cu_seqlens_q,
        cu_seqlens_k,
        alibi_ref,
        static_cast<int>(max_seqlen_q),
        static_cast<int>(max_seqlen_k),
        static_cast<float>(p_dropout),
        static_cast<float>(softmax_scale),
        zero_tensors,
        is_causal,
        static_cast<int>(window_left),
        static_cast<int>(window_right),
        static_cast<float>(softcap),
        deterministic,
        gen,
        rng_ref
    );
}

// fwd_kvcache
std::vector<at::Tensor> torchbind_fwd_kvcache(
    at::Tensor& q,
    const at::Tensor& kcache,
    const at::Tensor& vcache,
    std::optional<at::Tensor> k,
    std::optional<at::Tensor> v,
    std::optional<at::Tensor> seqlens_k,
    std::optional<at::Tensor> rotary_cos,
    std::optional<at::Tensor> rotary_sin,
    std::optional<at::Tensor> cache_batch_idx,
    std::optional<at::Tensor> leftpad_k,
    std::optional<at::Tensor> block_table,
    std::optional<at::Tensor> alibi_slopes,
    std::optional<at::Tensor> out,
    double  softmax_scale,
    bool    is_causal,
    int64_t window_left,
    int64_t window_right,
    double  softcap,
    bool    is_rotary_interleaved,
    int64_t num_splits
) {
    std::optional<const at::Tensor> k_ref          = convert_opt(k);
    std::optional<const at::Tensor> v_ref          = convert_opt(v);
    std::optional<const at::Tensor> seqlens_ref    = convert_opt(seqlens_k);
    std::optional<const at::Tensor> rotary_cos_ref = convert_opt(rotary_cos);
    std::optional<const at::Tensor> rotary_sin_ref = convert_opt(rotary_sin);
    std::optional<const at::Tensor> cache_idx_ref  = convert_opt(cache_batch_idx);
    std::optional<const at::Tensor> leftpad_ref    = convert_opt(leftpad_k);
    std::optional<at::Tensor> block_ref            = block_table;
    std::optional<at::Tensor> alibi_ref            = alibi_slopes;
    std::optional<at::Tensor> out_ref              = out;

    return flash_attention_kvcache(
        q,
        kcache,
        vcache,
        k_ref,
        v_ref,
        seqlens_ref,
        rotary_cos_ref,
        rotary_sin_ref,
        cache_idx_ref,
        leftpad_ref,
        block_ref,
        alibi_ref,
        out_ref,
        static_cast<float>(softmax_scale),
        is_causal,
        static_cast<int>(window_left),
        static_cast<int>(window_right),
        static_cast<float>(softcap),
        is_rotary_interleaved,
        static_cast<int>(num_splits)
    );
}

TORCH_LIBRARY(flash_attn_v100, m) {
    m.def(
        "fwd(Tensor(a!) q, Tensor k, Tensor v, "
        "Tensor? out, Tensor? alibi_slopes, "
        "float p_dropout, float softmax_scale, bool is_causal, "
        "int window_left, int window_right, float softcap, "
        "bool return_softmax, Generator? gen) -> Tensor[]",
        &torchbind_fwd);

    m.def(
        "bwd(Tensor dout, Tensor q, Tensor k, Tensor v, "
        "Tensor out, Tensor softmax_lse, "
        "Tensor? dq, Tensor? dk, Tensor? dv, Tensor? alibi_slopes, "
        "float p_dropout, float softmax_scale, bool is_causal, "
        "int window_left, int window_right, float softcap, "
        "bool deterministic, Generator? gen, Tensor? rng_state) -> Tensor[]",
        &torchbind_bwd);

    m.def(
        "varlen_fwd(Tensor(a!) q, Tensor k, Tensor v, Tensor? out, "
        "Tensor cu_seqlens_q, Tensor cu_seqlens_k, "
        "Tensor? seqused_k, Tensor? leftpad_k, "
        "Tensor? block_table, Tensor? alibi_slopes, "
        "int max_seqlen_q, int max_seqlen_k, "
        "float p_dropout, float softmax_scale, bool zero_tensors, bool is_causal, "
        "int window_left, int window_right, float softcap, "
        "bool return_softmax, Generator? gen, int num_splits) -> Tensor[]",
        &torchbind_varlen_fwd);

    m.def(
        "varlen_bwd(Tensor dout, Tensor q, Tensor k, Tensor v, "
        "Tensor out, Tensor softmax_lse, "
        "Tensor? dq, Tensor? dk, Tensor? dv, "
        "Tensor cu_seqlens_q, Tensor cu_seqlens_k, Tensor? alibi_slopes, "
        "int max_seqlen_q, int max_seqlen_k, "
        "float p_dropout, float softmax_scale, bool zero_tensors, bool is_causal, "
        "int window_left, int window_right, float softcap, "
        "bool deterministic, Generator? gen, Tensor? rng_state) -> Tensor[]",
        &torchbind_varlen_bwd);

    m.def(
        "fwd_kvcache(Tensor(a!) q, Tensor kcache, Tensor vcache, "
        "Tensor? k, Tensor? v, Tensor? seqlens_k, "
        "Tensor? rotary_cos, Tensor? rotary_sin, "
        "Tensor? cache_batch_idx, Tensor? leftpad_k, "
        "Tensor? block_table, Tensor? alibi_slopes, Tensor? out, "
        "float softmax_scale, bool is_causal, "
        "int window_left, int window_right, float softcap, "
        "bool is_rotary_interleaved, int num_splits) -> Tensor[]",
        &torchbind_fwd_kvcache);
}
