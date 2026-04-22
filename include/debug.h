// ============================================================================
// debug.h - Debug framework for Kernel
// ============================================================================
#pragma once

#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>

#ifndef KERNEL_DEBUG
    #define KERNEL_DEBUG 0
#endif

#if KERNEL_DEBUG
    #define __ASM_MARK(STG, CTX, TYPE, MAGIC) \
        do { \
            volatile unsigned int point = 0; \
            asm volatile("mov.u32 %0, " #MAGIC "; // DBG_PTX_" #STG "_" #CTX "_" #TYPE "\n\t" \
                         : "+r"(point) :: "memory"); \
            (void)point; \
        } while(0)

    #define __ASM_DEBUG_BEGIN(STG, CTX) __ASM_MARK(STG, CTX, BEGIN, 0xBEEF0001)
    #define __ASM_DEBUG_END(STG, CTX)   __ASM_MARK(STG, CTX, END,   0xCAFE0002)
#else
    #define __ASM_DEBUG_BEGIN(STG, CTX) ((void)0)
    #define __ASM_DEBUG_END(STG, CTX)   ((void)0)
#endif
