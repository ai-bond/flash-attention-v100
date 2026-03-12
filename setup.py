# *
# * Copyright (c) 2025, D.Skryabin / tg @ai_bond007
# * SPDX-License-Identifier: BSD-3-Clause
# *

import os
from pathlib import Path
from packaging.version import parse
from setuptools import setup

this_dir = Path(__file__).parent.resolve()

def get_ext_modules():
    try:
        from torch.utils.cpp_extension import CUDAExtension
    except ImportError as e:
        raise RuntimeError(
            "torch is required to build flash_attn_v100. "
            "Please install torch >= 2.5 first (e.g., `pip install torch --index-url https://download.pytorch.org/whl/cu118`)."
        ) from e

    return [
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
                    "-Wno-deprecated-gpu-targets",
                ],
            },
        )
    ]

def get_cmdclass():
    try:
        from torch.utils.cpp_extension import BuildExtension
    except ImportError as e:
        raise RuntimeError(
            "torch is required to build flash_attn_v100. "
            "Please install torch >= 2.5 first."
        ) from e

def get_cmdclass():
    try:
        from torch.utils.cpp_extension import BuildExtension
    except ImportError as e:
        raise RuntimeError(
            "torch is required to build flash_attn_v100. "
            "Please install torch >= 2.5 first."
        ) from e

    class CustomBuildExtension(BuildExtension):
        def build_extensions(self):
            import torch

            if not os.environ.get("MAX_JOBS"):
                try:
                    import psutil

                    nvcc_threads = int(os.environ.get("NVCC_THREADS", 3))
                    cores = os.cpu_count() or 4
                    free_mem_gb = psutil.virtual_memory().available / (1024**3)

                    mem_per_job_gb = 2.5

                    max_jobs_mem = max(1, int(free_mem_gb / mem_per_job_gb))
                    max_jobs_cores = max(1, (cores - 2) // nvcc_threads)

                    jobs = max(1, min(max_jobs_mem, max_jobs_cores, 6))

                    if free_mem_gb >= 10 and cores >= 8:
                        jobs = max(jobs, 4)

                    os.environ["MAX_JOBS"] = str(jobs)
                    os.environ["NVCC_THREADS"] = str(nvcc_threads)

                    print(f"autoset max_jobs={jobs}, nvcc_threads={nvcc_threads} "
                          f"(current {cores} cores, {free_mem_gb:.1f}GB free mem)")
                except Exception as e:
                    print(f"Warning: could not auto-tune build params: {e}")
                    pass

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required but not available.")
            if parse(torch.version.cuda) < parse("11.6"):
                raise RuntimeError(f"CUDA version {torch.version.cuda} < 11.6 is not supported. Please use CUDA ≥ 11.6 (e.g., PyTorch built with CUDA 11.8/12.x).")
            super().build_extensions()

    return {"build_ext": CustomBuildExtension}

try:
    with open(this_dir / "README.md", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Flash Attention implementation for Tesla V100"

setup(
    name="flash_attn_v100",
    version="25.11",
    packages=["flash_attn_v100"],
    ext_modules=get_ext_modules(),
    cmdclass=get_cmdclass(),
    python_requires=">=3.10",
    zip_safe=False,
    author="D.Skryabin",
    author_email="tg @ai_bond007",
    description="Flash Attention implementation under unsupported Tesla V100",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ai-bond/flash-attention-v100",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
    ],
)