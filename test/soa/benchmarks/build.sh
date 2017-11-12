#!/bin/bash
set -e
echo "Script will stop on error or incorrect result."

mkdir -p bin
mkdir -p assembly

for v_compiler in "g++" "clang++-3.8"
do
  for v_storage in "StaticStorage" "DynamicStorage"
  do
    for v_opt_mode in "-O0" "-O3"
    do
      for v_addr_mode in "0" "4"
      do
        out_name="${v_compiler}_${v_opt_mode}_${v_storage}_${v_addr_mode}"
        ${v_compiler} -std=c++11 ${v_opt_mode} \
            -DSTORAGE_STRATEGY=${v_storage} \
            -DADDRESS_MODE=${v_addr_mode} \
            codegen_test.cc \
            -I../../../ikra \
            -o bin/${out_name}
        bin/${out_name}
        objdump -S bin/${out_name} > assembly/${out_name}.S
      done
    done
  done
done

# CUDA test
/usr/local/cuda/bin/nvcc \
    -std=c++14 \
    --expt-extended-lambda \
    -O3 \
    -I../../../ikra \
    -o bin/cuda_codegen_test \
    cuda_codegen_test.cu
bin/cuda_codegen_test
/usr/local/cuda/bin/cuobjdump bin/cuda_codegen_test -ptx -sass -res-usage \
    > assembly/cuda_codegen_test.S


for v_compiler in "g++" "clang++-3.8"
do
  # Build nbody
  out_name="${v_compiler}_nbody_ikracpp"
  ${v_compiler} -O3 nbody/ikracpp.cc -std=c++11 -I../../../ikra \
      -o bin/${out_name}
  objdump -S bin/${out_name} > assembly/${out_name}.S

  out_name="${v_compiler}_nbody_ikracpp_field"
  ${v_compiler} -O3 nbody/ikracpp_field.cc -std=c++11 -I../../../ikra \
      -o bin/${out_name}
  objdump -S bin/${out_name} > assembly/${out_name}.S

  # No vectorization with: -fno-tree-vectorize
  out_name="${v_compiler}_nbody_soa"
  ${v_compiler} -O3 nbody/soa.cc -std=c++11 -o bin/${out_name}
  objdump -S bin/${out_name} > assembly/${out_name}.S

  out_name="${v_compiler}_nbody_aos"
  ${v_compiler} -O3 nbody/aos.cc -std=c++11 -o bin/${out_name}
  objdump -S bin/${out_name} > assembly/${out_name}.S
done
