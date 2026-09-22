#!/bin/bash

if [ "$1" = "-h" ] || [ "$1" = "--help" ] || [ $# -eq 0 ]; then
    echo "Usage: $0 [--native] [--volta] [--delete]"
    echo ""
    echo "Options:"
    echo "  --native   Compile with mma.h (MMA_NATIVE)"
    echo "  --volta    Compile with mma_m16n16k16.h (USE_VOLTA_MMA)"
    echo "  --delete   Clean compiled binaries and PTX/SASS files"
    echo ""
    echo "Examples:"
    echo "  $0 --native          # mma.h"
    echo "  $0 --volta           # mma_m16n16k16.h"
    echo "  $0 --delete          # Clean all"
    echo "  $0 --volta --delete  # Clean volta files only"
    exit 0
fi

# Resolve include/ relative to this script, so cwd doesn't matter
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INC="-I${SCRIPT_DIR}/../../../include"

USE_VOLTA=0
USE_NATIVE=0
DO_CLEAN=0
DO_COMPILE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --native)  USE_NATIVE=1; DO_COMPILE=1; shift ;;
        --volta)   USE_VOLTA=1;  DO_COMPILE=1; shift ;;
        --delete|--clean) DO_CLEAN=1; shift ;;
        *)
            echo "Unknown option: $1"
            echo "Use: $0 --native | --volta | --delete"
            exit 1
            ;;
    esac
done

if [ $USE_VOLTA -eq 1 ]; then
    SUFFIX="_volta"
    DEFINE_FLAG="-DUSE_VOLTA_MMA"
elif [ $USE_NATIVE -eq 1 ]; then
    SUFFIX="_native"
    DEFINE_FLAG="-DMMA_NATIVE"
else
    SUFFIX=""
    DEFINE_FLAG=""
fi

if [ $DO_CLEAN -eq 1 ]; then
    echo "Cleaning..."
    for cu in *.cu; do
        [ -f "$cu" ] || continue
        b="${cu%.cu}"
        if [ $USE_VOLTA -eq 1 ]; then
            rm -f "${b}_volta" "${b}_volta.ptx" "${b}_volta.sass" "${b}_volta.cubin"
            echo "  Removed ${b}_volta{,.ptx,.sass,.cubin}"
        elif [ $USE_NATIVE -eq 1 ]; then
            rm -f "${b}_native" "${b}_native.ptx" "${b}_native.sass" "${b}_native.cubin"
            echo "  Removed ${b}_native{,.ptx,.sass,.cubin}"
        else
            rm -f "$b" "${b}.ptx" "${b}.sass" "${b}.cubin" \
                  "${b}_volta" "${b}_volta.ptx" "${b}_volta.sass" "${b}_volta.cubin" \
                  "${b}_native" "${b}_native.ptx" "${b}_native.sass" "${b}_native.cubin"
            echo "  Removed $b and all variants"
        fi
    done
    echo "Done."
fi

if [ $DO_COMPILE -eq 1 ]; then
    for cu in *.cu; do
        [ -f "$cu" ] || continue
        b="${cu%.cu}"
        out="${b}${SUFFIX}"
        ptx="${b}${SUFFIX}.ptx"
        cubin="${b}${SUFFIX}.cubin"
        sass="${b}${SUFFIX}.sass"

        # Executable
        if [ ! -f "$out" ]; then
            echo "Compile object $cu -> $out"
            nvcc -arch=sm_70 -O3 -lineinfo -Wno-deprecated-gpu-targets $INC $DEFINE_FLAG "$cu" -o "$out"
        else
            echo "Already compiled $cu -> $out ... skip"
        fi

        # Full PTX
        if [ ! -f "$ptx" ]; then
            echo "Compile ptx $cu -> $ptx"
            nvcc -arch=sm_70 -lineinfo -ptx -Wno-deprecated-gpu-targets $INC $DEFINE_FLAG "$cu" -o "$ptx"
        else
            echo "Already compiled $cu -> $ptx ... skip"
        fi

        # Full SASS via cubin + cuobjdump
        if [ ! -f "$sass" ]; then
            echo "Compile cubin $cu -> $cubin"
            nvcc -arch=sm_70 -cubin -lineinfo -Wno-deprecated-gpu-targets $INC $DEFINE_FLAG "$cu" -o "$cubin"
            echo "Dump sass $cubin -> $sass"
            cuobjdump -sass "$cubin" > "$sass"
        else
            echo "Already dumped $cu -> $sass ... skip"
        fi
    done
fi