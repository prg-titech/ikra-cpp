#!/bin/bash
set -e

mkdir -p bin
rm -f result_simple_cuda.txt

runs_aos=4
runs=12
echo "v_size: irka*${runs}, soa*${runs}, aos*${runs_aos}"

for v_size in `./size_generator_cuda.py`
do
  /usr/local/cuda/bin/nvcc \
      -std=c++11 \
      --expt-extended-lambda \
      -O3 \
      -DkNumBodies=${v_size} \
      -I../../../../../ikra \
      -o bin/cuda_nbody_soa \
      simple_cuda_soa.cu
  echo -n "."

  /usr/local/cuda/bin/nvcc \
      -std=c++11 \
      --expt-extended-lambda \
      -O3 \
      -DkNumBodies=${v_size} \
      -I../../../../../ikra \
      -o bin/cuda_nbody_aos \
      simple_cuda_aos.cu
  echo -n "."

  /usr/local/cuda/bin/nvcc \
      -std=c++11 \
      --expt-extended-lambda \
      -O3 \
      -DkNumBodies=${v_size} \
      -I../../../../../ikra \
      -o bin/cuda_nbody_ikra \
      simple_cuda_ikra.cu
  echo -n "."

  runs=`./size_generator_cuda_runs.py ${v_size}`
  line="${runs},${v_size}"

  for ((i=0; i<$runs; i++))
  do
    result=`bin/cuda_nbody_ikra`
    line="${line},${result}"
    echo -n "."
  done

  for ((i=0; i<$runs; i++))
  do
    result=`bin/cuda_nbody_soa`
    line="${line},${result}"
    echo -n "."
  done

  for ((i=0; i<$runs; i++))
  do
    result=`bin/cuda_nbody_aos`
    line="${line},${result}"
    echo -n "."
  done

  echo $line >> result_simple_cuda.txt
  echo "   ---> $line"
done
