#!/bin/bash
set -e
echo "Script will stop on error or incorrect result."
extra_args="-march=native -fomit-frame-pointer"
clang_bin="clang++-5.0"

mkdir -p bin
mkdir -p assembly

for v_compiler in "g++" "${clang_bin}"
do
  for v_storage in "StaticStorage" "DynamicStorage"
  do
    for v_opt_mode in "-O0" "-O3"
    do
      for v_addr_mode in "0" "4"
      do
        out_name="${v_compiler}_${v_opt_mode}_${v_storage}_${v_addr_mode}_Soa"
        ${v_compiler} -std=c++11 ${v_opt_mode} ${extra_args} \
            -DSTORAGE_STRATEGY=${v_storage} \
            -DADDRESS_MODE=${v_addr_mode} \
            -DLAYOUT_MODE=Soa \
            codegen_test.cc \
            -I../../../ikra \
            -o bin/${out_name}
        bin/${out_name}
        objdump -S bin/${out_name} > assembly/${out_name}.S
      done
    done
  done
done

for v_compiler in "g++" "${clang_bin}"
do
  for v_opt_mode in "-O0" "-O3"
  do
    for v_layout_mode in "Aos" "Soa"
    do
      out_name="${v_compiler}_${v_opt_mode}_StaticStorage_0_Aos"
      ${v_compiler} -std=c++11 ${v_opt_mode} ${extra_args} \
          -DSTORAGE_STRATEGY=StaticStorage \
          -DADDRESS_MODE=0 \
          -DLAYOUT_MODE=Aos \
          codegen_test.cc \
          -I../../../ikra \
          -o bin/${out_name}
      bin/${out_name}
      objdump -S bin/${out_name} > assembly/${out_name}.S
    done
  done
done


# CUDA test
/usr/local/cuda/bin/nvcc \
    -std=c++11 \
    --expt-extended-lambda \
    -O3 \
    -I../../../ikra \
    -o bin/cuda_codegen_test \
    cuda_codegen_test.cu
bin/cuda_codegen_test
/usr/local/cuda/bin/cuobjdump bin/cuda_codegen_test -ptx -sass -res-usage \
    > assembly/cuda_codegen_test.S

g++ -std=c++11 -O3 -I../../../ikra -o bin/cuda_aos_style_cpu array/aos_style_cpu.cc
bin/cuda_aos_style_cpu

/usr/local/cuda/bin/nvcc \
    -std=c++11 \
    --expt-extended-lambda \
    -O3 \
    -I../../../ikra \
    -o bin/cuda_array_aos_style \
    array/aos_style.cu
bin/cuda_array_aos_style
/usr/local/cuda/bin/cuobjdump bin/cuda_array_aos_style -ptx -sass -res-usage \
    > assembly/cuda_array_aos_style.S

/usr/local/cuda/bin/nvcc \
    -std=c++11 \
    --expt-extended-lambda \
    -O3 \
    -I../../../ikra \
    -o bin/cuda_array_soa_style \
    array/soa_style.cu
bin/cuda_array_soa_style
/usr/local/cuda/bin/cuobjdump bin/cuda_array_soa_style -ptx -sass -res-usage \
    > assembly/cuda_array_soa_style.S

/usr/local/cuda/bin/nvcc \
    -std=c++11 \
    --expt-extended-lambda \
    -O3 \
    -I../../../ikra \
    -o bin/cuda_array_inlined_soa_style \
    array/inlined_soa_style.cu
bin/cuda_array_inlined_soa_style
/usr/local/cuda/bin/cuobjdump bin/cuda_array_inlined_soa_style -ptx -sass -res-usage \
    > assembly/cuda_array_inlined_soa_style.S

extra_args="-march=native -fomit-frame-pointer -Ofast"
for v_compiler in "g++" "${clang_bin}"
do
  # Build nbody
  out_name="${v_compiler}_nbody_ikracpp"
  ${v_compiler} -O3 ${extra_args} nbody/ikracpp.cc -std=c++11 -I../../../ikra \
      -o bin/${out_name}
  objdump -S bin/${out_name} > assembly/${out_name}.S
  echo -n "."

  out_name="${v_compiler}_nbody_ikracpp_field"
  ${v_compiler} -O3 ${extra_args} nbody/ikracpp_field.cc -std=c++11 -I../../../ikra \
      -o bin/${out_name}
  objdump -S bin/${out_name} > assembly/${out_name}.S
  echo -n "."

  out_name="${v_compiler}_nbody_ikracpp_inversed"
  ${v_compiler} -O3 ${extra_args} nbody/ikracpp_inversed.cc -std=c++11 -I../../../ikra \
      -o bin/${out_name}
  objdump -S bin/${out_name} > assembly/${out_name}.S
  echo -n "."

  # No vectorization with: -fno-tree-vectorize
  out_name="${v_compiler}_nbody_soa"
  ${v_compiler} -O3 ${extra_args} nbody/soa.cc -std=c++11 -o bin/${out_name}
  objdump -S bin/${out_name} > assembly/${out_name}.S
  echo -n "."

  out_name="${v_compiler}_nbody_aos"
  ${v_compiler} -O3 ${extra_args} nbody/aos.cc -std=c++11 -o bin/${out_name}
  objdump -S bin/${out_name} > assembly/${out_name}.S
  echo -n "."
done

/usr/local/cuda/bin/nvcc \
    -std=c++11 \
    --expt-extended-lambda \
    -O3 \
    -I../../../ikra \
    -o bin/cuda_nbody_ikracpp_inversed_gpu \
    nbody/ikracpp_inversed_gpu.cu
/usr/local/cuda/bin/cuobjdump bin/cuda_nbody_ikracpp_inversed_gpu \
    -ptx -sass -res-usage > assembly/cuda_nbody_ikracpp_inversed_gpu.S
echo -n "."

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
