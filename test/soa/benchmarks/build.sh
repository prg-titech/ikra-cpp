#!/bin/bash
set -e
echo "Script will stop on error or incorrect result."
extra_args="-march=native -fomit-frame-pointer"
clang_bin="clang++-3.8"

mkdir -p bin
mkdir -p assembly

/usr/local/cuda/bin/nvcc \
    -std=c++11 \
    --expt-extended-lambda \
    -O3 \
    -I../../../ikra \
    -o bin/cuda_nbody_soa_inversed_gpu \
    nbody/soa_inversed.cu
/usr/local/cuda/bin/cuobjdump bin/cuda_nbody_soa_inversed_gpu \
    -ptx -sass -res-usage > assembly/cuda_nbody_soa_inversed_gpu.S
echo -n "."

/usr/local/cuda/bin/nvcc \
    -std=c++11 \
    --expt-extended-lambda \
    -O3 \
    -I../../../ikra \
    -o bin/cuda_nbody_ikracpp_inversed_field_gpu \
    nbody/ikracpp_inversed_field_gpu.cu
/usr/local/cuda/bin/cuobjdump bin/cuda_nbody_ikracpp_inversed_field_gpu \
    -ptx -sass -res-usage > assembly/cuda_nbody_ikracpp_inversed_field_gpu.S
echo "."
