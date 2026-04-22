# *
# * Copyright (c) 2025, D.Skryabin / tg @ai_bond007
# * SPDX-License-Identifier: BSD-3-Clause
# *

import os
import shutil
from pathlib import Path
from packaging.version import parse
from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.install import install
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

this_dir = Path(__file__).parent.resolve()

def get_ext_modules():
    try:
        from torch.utils.cpp_extension import CUDAExtension
    except ImportError as e:
        raise RuntimeError(
            "torch is required to build flash_attn_v100. "
            "Please install torch >= 2.10 first (e.g., `pip install torch==2.10.0+cu129 --index-url https://download.pytorch.org/whl/cu129`)."
        ) from e

    nvcc_threads = int(os.environ.get("NVCC_THREADS", 4))
    nvcc_flags = [
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
        f"--threads={nvcc_threads}",
    ]

    if os.environ.get("ATTENTION_DEBUG"):
        nvcc_flags.extend([
            "-g",
            "-Xptxas", "-v",
            "--keep",
            "--keep-dir", str(this_dir / "build"),
        ])
        (this_dir / "build").mkdir(exist_ok=True)

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
                "nvcc": nvcc_flags,
            },
        )
    ]

class CopyAttention(build_py):
    def run(self):
        super().run()
        build_lib = self.build_lib
        this_dir = os.path.dirname(os.path.abspath(__file__))

        src_pkg = os.path.join(this_dir, 'flash_attn')
        dst_pkg = os.path.join(build_lib, 'flash_attn')
        if os.path.exists(src_pkg):
            if os.path.exists(dst_pkg):
                shutil.rmtree(dst_pkg)
            shutil.copytree(src_pkg, dst_pkg, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.so'))
            print(f"Copied package: {src_pkg} -> {dst_pkg}")

class InstallAttention(install):
    def run(self):
        install.run(self)
        import site
        dst = os.path.join(site.getsitepackages()[0], 'flash_attn-2.8.3.dist-info')
        if os.path.exists(dst): shutil.rmtree(dst)
        os.makedirs(dst, exist_ok=True)
        with open(os.path.join(dst, 'METADATA'), 'w') as f:
            f.write("Metadata-Version: 2.4\nName: flash-attn\nVersion: 2.8.3\n")
        with open(os.path.join(dst, 'top_level.txt'), 'w') as f:
            f.write('flash_attn\n')

def get_cmdclass():
    try:
        from torch.utils.cpp_extension import BuildExtension
    except ImportError as e:
        raise RuntimeError(
            "torch is required to build flash_attn_v100. "
            "Please install torch >= 2.10 first (e.g., `pip install torch==2.10.0+cu129 --index-url https://download.pytorch.org/whl/cu129`)."
        ) from e

    class BuildAttention(BuildExtension):
        def build_extensions(self):
            import torch

            if not os.environ.get("MAX_JOBS"):
                try:
                    import psutil

                    nvcc_threads = int(os.environ.get("NVCC_THREADS", 4))
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
            if parse(torch.version.cuda) < parse("12.9"):
                raise RuntimeError(f"CUDA version {torch.version.cuda} < 12.9 is not supported.")
            super().build_extensions()

    return {"build_ext": BuildAttention, "build_py": CopyAttention, 'install': InstallAttention}

try:
    with open(this_dir / "README.md", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Flash Attention implementation for Tesla V100"

setup(
    name="flash_attn_v100",
    version="26.04",
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