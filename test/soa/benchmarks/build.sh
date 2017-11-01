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
