# FlashAttention for unsupported Tesla v100

This repository want to implement the official implementation of FlashAttention and [FlashAttention-2](https://github.com/ai-bond/flash-attention-v100/blob/main/docs/attention.md) under unsupported in TriDao repo [Nvidia Tesla V100](https://github.com/ai-bond/flash-attention-v100/blob/main/docs/volta.md)

> This repo is attempt to build flash attention from scratch without "Vibe Code" for self education. 

According to [Nvidia Deprecated Architectures](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#deprecated-architectures): Architecture support for Volta is considered feature-complete. Offline compilation and library support for these architectures have been removed in CUDA Toolkit 13.0 major version release.

Last one available CUDA for Volta:
-------------
```
# Download package
wget https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run

# Install, this cuda package with NVIDIA driver version
#      575.57.08 that can be installed together
sudo sh cuda_12.9.1_575.57.08_linux.run

# Export and apply
cat >> ~/.bashrc << 'EOF'
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
EOF
source ~/.bashrc
```


Deployment and compilation
-------------

```bash
# Create new python virtual env or use own existed:
python -m venv env
source env/bin/activate

# Update pip
pip install --upgrade pip

# Clone code and install packages:
git clone https://github.com/ai-bond/flash-attention-v100/
cd ./flash-attention-v100

# Install req packages
pip install -r requirements.txt
```
As NVIDIA deprecated Volta support in CUDA since viersion 13 then PyTorch also restrict and deprecated support in new versions:  [PyTorch is dropping Volta support from CUDA-12.8 binaries for release 2.11](https://dev-discuss.pytorch.org/t/dropping-volta-support-from-cuda-12-8-binaries-for-release-2-11/) and check [PyTorch \[release 2.8-2.9\] delete support for Maxwell, Pascal, and Volta architectures for CUDA 12.8 and 12.9 builds](https://github.com/pytorch/pytorch/issues/157517)

```bash
# Install last one PyTorch that's support with 12.9 CUDA
pip install torch==2.10.0+cu129 --index-url https://download.pytorch.org/whl/cu129

# Check is package supports Volta
python -c "import torch; p=torch.cuda.get_device_properties(0); print(f'{p.name} SM {p.major}.{p.minor} supported')"

# If you will see Tesla V100-XXX-XXGB SM 7.0 supported all is done.
# We can compile and install project with just:

./run.sh 

or 

pip install . --no-build-isolation -v
```
Also after

```
Successfully built flash_attn_v100
Installing collected packages: flash_attn_v100
Successfully installed flash_attn_v100-XX.XX

# just check exactly flash_attn import thru

python -c 'import flash_attn; print(f"Version: {flash_attn.__doc__}")'
Should: Flash Attention for Tesla V100 v2.8.3

and

pip show flash_attn
Name: flash-attn
Version: 2.8.3
Summary: Flash Attention for Tesla V100

```

And gl and hf :)

Debug
-------------
Now by default code will compile with fused m16n16k16 library. Youcan back or use

`--mma-native` Use native CUDA mma.h with (16x16x16, m32n8k16, m8n32k16)

`--mma-884` Use fused 8x8x4 MMA library

 `--debug` Enable debug mode, preserve build artifacts, generate assembly extraction tool



## Debug Markers

### Available Stages

| Stage | Description | Location |
|-------|-------------|----------|
| `SMEM` | Shared memory initialization check | After smem init, before any loads |
| `TILE` | Tile load validation (Q, K, V) | After each tile load from global memory |
| `SQKT` | Score matrix (Q·K^T) validation | After matrix multiplication, before softmax |
| `SOFTMAX` | Online softmax validation | After softmax computation |
| `DOVT` | dO·V^T validation (backward) | After gradient computation for dP |
| `DOPV` | Output accumulation (P·V) validation | After each P·V multiplication |
| `DQDSK` | dQ from dS·K validation (backward) | After dQ gradient computation |
| `DVPTDO` | dV from P^T·dO validation (backward) | After dV gradient computation |
| `DKDSTQ` | dK from S^T·dQ validation (backward) | After dK gradient computation |
| `ROWDQ` | Row-wise operations for dQ | During dQ reduction |
| `ROWDKV` | Row-wise operations for dK/dV | During dK/dV reduction |
| `WRITEO` | Final output write validation | Before writing O to global memory |
| `WRITEQ` | dQ write validation (backward) | Before writing dQ to global memory |
| `WRITEKV` | dK/dV write validation (backward) | Before writing dK/dV to global memory |
| `NONE` | No debug stage | Default/disabled state |

### Extracted Checks

- `__CHECK_INIT`      - Verifies buffer zeroing (Q, K, O)
- `__CHECK_ERRORS`    - Detects inf/nan in shared memory matrices
- `__PRINT_MATRIX`    - Dumps matrix content (optional)
- `__PRINT_RESULT`    - Traces row sums for online softmax
- `__ASM_DEBUG_BEGIN` - Put begin of ptx and sass insertion
- `__ASM_DEBUG_BEGIN` - Put end of ptx and sass insertion

When running with `--debug` flag, the kernels insert assembly-level markers for PTX/SASS extraction using `asm_extract.sh` script.

```bash
# Extract PTX block
./build/asm_extract.sh ./build/fused_mha_forward.ptx SQKT ptx

# Extract SASS block  
./build/asm_extract.sh ./build/fused_mha_backward.cubin DOPV sass

