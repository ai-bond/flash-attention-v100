#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 [--compile] [--volta] [--delete]"
    echo ""
    echo "Options:"
    echo "  --compile  Compile with mma.h (standard)"
    echo "  --volta    Compile with fused_mma_m16n16k16.h"
    echo "  --delete   Clean compiled binaries and PTX files"
    echo ""
    echo "Examples:"
    echo "  $0 --compile         # Compile with mma.h"
    echo "  $0 --volta           # Compile with volta header"
    echo "  $0 --delete          # Clean all"
    echo "  $0 --volta --delete  # Clean volta files only"
    exit 0
fi

USE_VOLTA=0
DO_CLEAN=0
DO_COMPILE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --compile)
            DO_COMPILE=1
            shift
            ;;
        --volta)
            USE_VOLTA=1
            DO_COMPILE=1
            shift
            ;;
        --delete|--clean)
            DO_CLEAN=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use: $0 --compile | --volta | --delete"
            exit 1
            ;;
    esac
done

if [ $DO_CLEAN -eq 1 ]; then
    echo "Cleaning..."
    for cu_file in *.cu; do
        if [ -f "$cu_file" ]; then
            base_name="${cu_file%.cu}"

            if [ $USE_VOLTA -eq 1 ]; then
                rm -f "${base_name}_volta" "${base_name}_volta.ptx"
                echo "  Removed ${base_name}_volta, ${base_name}_volta.ptx"
            else
                rm -f "$base_name" "${base_name}.ptx" "${base_name}_volta" "${base_name}_volta.ptx"
                echo "  Removed $base_name, ${base_name}.ptx, ${base_name}_volta, ${base_name}_volta.ptx"
            fi
        fi
    done
    echo "Done."
fi

if [ $DO_COMPILE -eq 1 ]; then
    for cu_file in *.cu; do
        if [ -f "$cu_file" ]; then
            base_name="${cu_file%.cu}"

            if [ $USE_VOLTA -eq 1 ]; then
                out_name="${base_name}_volta"
                ptx_name="${base_name}_volta.ptx"
                define_flag="-DUSE_VOLTA_MMA"
            else
                out_name="${base_name}"
                ptx_name="${base_name}.ptx"
                define_flag=""
            fi

            if [ ! -f "$out_name" ]; then
                echo "Compile object $cu_file -> $out_name"
                nvcc -arch=sm_70 -O3 -Wno-deprecated-gpu-targets $define_flag "$cu_file" -o "$out_name"
            else
                echo "Already compiled $cu_file -> $out_name ... skip"
            fi

            if [ ! -f "$ptx_name" ]; then
                echo "Compile ptx $cu_file -> $ptx_name"
                nvcc -arch=sm_70 -O3 -ptx -Wno-deprecated-gpu-targets $define_flag "$cu_file" -o "$ptx_name"
            else
                echo "Already compiled $cu_file -> $ptx_name ... skip"
            fi
        fi
    done
fi