# *
# * Copyright (c) 2025, D.Skryabin / tg @ai_bond007
# * SPDX-License-Identifier: BSD-3-Clause
# *

import os
from pathlib import Path
from packaging.version import parse

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if torch.cuda.is_available():
    if parse(torch.version.cuda) < parse("11.6"):
        raise RuntimeError("CUDA version should be ≥ 11.6")
else:
    raise RuntimeError("CUDA unavailable")

this_dir = Path(__file__).parent.resolve()

open(this_dir / "README.md", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Flash Attention implementation for Tesla V100"

ext_modules = [
    CUDAExtension(
        name="flash_attn_v100_cuda",
        sources=[
            "kernel/fused_mha_api.cpp",
            "kernel/fused_mha_forward.cu",
            "kernel/fused_mha_backward.cu",
        ],
        include_dirs=[this_dir / "include"],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": [
                "-O3",
                "-std=c++17",
                "-gencode", "arch=compute_70,code=sm_70",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
            ],
        },
    )
]

# Установка
setup(
    name="flash_attn_v100",
    version="1.2.0",
    packages=[],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(parallel=True, use_ninja=True)},
    python_requires=">=3.10",
    install_requires=["torch>=2.5"],
    zip_safe=False,
    author="D.Skryabin",
    author_email="tg @ai_bond007",
    description="Flash Attention implementation under unsupported Tesla V100",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ai-bond/flash-attention-v100",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
),
