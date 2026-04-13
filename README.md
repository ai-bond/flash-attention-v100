## FlashAttention for unsupported Tesla v100

This repository want to implement the official implementation of FlashAttention and [FlashAttention-2](https://github.com/ai-bond/flash-attention-v100/blob/main/docs/attention.md) under unsupported in TriDao repo [Nvidia Tesla V100](https://github.com/ai-bond/flash-attention-v100/blob/main/docs/volta.md)

### This repo is attempt to build flash attention from scratch without "Vibe Code" for self education. 

According to [Nvidia Deprecated Architectures](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#deprecated-architectures):

Architecture support for Volta is considered feature-complete. Offline compilation and library support for these architectures have been removed in CUDA Toolkit 13.0 major version release.

#### Last one available CUDA for Volta:
```
# Download package
wget https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run

# Install
sudo sh cuda_12.9.1_575.57.08_linux.run

# Export and apply
echo -e 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH\nexport PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```
This cuda package with NVIDIA driver version 575.57.08 that can be installed together at once.

#### Deployment and compilation

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
As NVIDIA deprecated Volta support in CUDA since viersion 13 then PyTorch also restrict and deprecated support in new versions.

[PyTorch is dropping Volta support from CUDA-12.8 binaries for release 2.11](https://dev-discuss.pytorch.org/t/dropping-volta-support-from-cuda-12-8-binaries-for-release-2-11/)

[PyTorch \[release 2.8-2.9\] delete support for Maxwell, Pascal, and Volta architectures for CUDA 12.8 and 12.9 builds](https://github.com/pytorch/pytorch/issues/157517)

```bash
# Install last one PyTorch that's support with 12.9 CUDA
pip install torch==2.10.0+cu129 --index-url https://download.pytorch.org/whl/cu129

# Check is package supports Volta
python -c "import torch; p=torch.cuda.get_device_properties(0); print(f'{p.name} SM {p.major}.{p.minor} supported')"
```
If you will see Tesla V100-XXX-XXGB SM 7.0 supported then all done and we can compile and install project with just:

```
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