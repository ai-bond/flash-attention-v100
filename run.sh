#!/bin/bash

clear
rm -rf ./build
python setup.py install && CUDA_LAUNCH_BLOCKING=1 python test.py


