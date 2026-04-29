// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ======================================================================================
// PHILOX 4x32_10: grid-stride deterministic
// ======================================================================================
__forceinline__ __device__ uint2 mulhilo32(unsigned int a, unsigned int b) {
    unsigned long long product;
    asm volatile("mul.wide.u32 %0, %1, %2;" : "=l"(product) : "r"(a), "r"(b));
    return reinterpret_cast<uint2&>(product);
}

__forceinline__ __device__ uint4 philox_single_round(const uint4 ctr, const uint2 key) {
    constexpr uint32_t PHILOX_M_A = 0xD2511F53U;
    constexpr uint32_t PHILOX_M_B = 0xCD9E8D57U;

    uint2 lo_prod = mulhilo32(PHILOX_M_A, ctr.x);
    uint2 hi_prod = mulhilo32(PHILOX_M_B, ctr.z);

    return make_uint4(
        hi_prod.y ^ ctr.y ^ key.x,
        hi_prod.x,
        lo_prod.y ^ ctr.w ^ key.y,
        lo_prod.x
    );
}

struct PhiloxState {
    uint2 key;
    uint4 counter;

    __device__ __forceinline__ uint4 next() {
        constexpr uint32_t KEY_STEP_A = 0x9E3779B9U;
        constexpr uint32_t KEY_STEP_B = 0xBB67AE85U;

        uint2 r_key = key;
        uint4 r_ctr = counter;

        #pragma unroll
        for (int round = 0; round < 9; ++round) {
            r_ctr = philox_single_round(r_ctr, r_key);
            r_key.x += KEY_STEP_A;
            r_key.y += KEY_STEP_B;
        }
        uint4 out_vec = philox_single_round(r_ctr, r_key);

        ++counter.x;
        if (counter.x == 0) {
            ++counter.y;
            if (counter.y == 0) {
                ++counter.z;
                if (counter.z == 0) ++counter.w;
            }
        }
        return out_vec;
    }
};

__device__ __forceinline__ PhiloxState init_philox(uint64_t seed, uint64_t offset) {
    PhiloxState st;
    st.key = reinterpret_cast<uint2&>(seed);
    st.counter.x = static_cast<uint32_t>(offset & 0xFFFFFFFFULL);
    st.counter.y = static_cast<uint32_t>(offset >> 32);
    st.counter.z = 0U;
    st.counter.w = 0U;
    return st;
}
