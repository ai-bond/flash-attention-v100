#!/bin/bash

clear
rm -rf ./build
pip install . --no-build-isolation -v && CUDA_LAUNCH_BLOCKING=1 python test.py


