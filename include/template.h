// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#pragma once

#include <type_traits>

// ======================================================================================
// Runtime / Compile-Time Dispatcher
// ======================================================================================
template <typename LaunchFn>
__host__ inline void dispatch_attention_features(
    bool is_causal, bool is_alibi, bool is_softcap, bool is_window, bool is_dropout,
    LaunchFn&& launch)
{
    const bool call_alibi   = is_causal && is_alibi;
    const bool call_softcap = is_causal && is_softcap;
    const bool call_window  = is_causal && is_window;
    const bool call_dropout = is_dropout;

    const int mask = (call_alibi ? 1 : 0) | (call_softcap ? 2 : 0) | (call_window ? 4 : 0) | (call_dropout ? 8 : 0);

    #define CALL(C, A, S, W, D) \
        launch(std::integral_constant<bool, C>{}, \
               std::integral_constant<bool, A>{}, \
               std::integral_constant<bool, S>{}, \
               std::integral_constant<bool, W>{}, \
               std::integral_constant<bool, D>{})

    if (!is_causal) {
        if (call_dropout) CALL(false, false, false, false, true);
        else              CALL(false, false, false, false, false);
    } else {
        switch (mask) {
            case 0:  CALL(true, false, false, false, false); break;
            case 1:  CALL(true, true,  false, false, false); break;
            case 2:  CALL(true, false, true,  false, false); break;
            case 3:  CALL(true, true,  true,  false, false); break;
            case 4:  CALL(true, false, false, true,  false); break;
            case 5:  CALL(true, true,  false, true,  false); break;
            case 6:  CALL(true, false, true,  true,  false); break;
            case 7:  CALL(true, true,  true,  true,  false); break;
            case 8:  CALL(true, false, false, false, true);  break;
            case 9:  CALL(true, true,  false, false, true);  break;
            case 10: CALL(true, false, true,  false, true);  break;
            case 11: CALL(true, true,  true,  false, true);  break;
            case 12: CALL(true, false, false, true,  true);  break;
            case 13: CALL(true, true,  false, true,  true);  break;
            case 14: CALL(true, false, true,  true,  true);  break;
            case 15: CALL(true, true,  true,  true,  true);  break;
        }
    }
    #undef CALL
}
