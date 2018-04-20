#!/bin/bash
# MODES: object == -1
#        fully inlined == -2

# Change directory to root.
cd "$(dirname "$0")"
cd ../..

function build_and_run() {
  cmake -DCMAKE_BUILD_TYPE=Release \
        -DBENCHMARK_MODE=1 \
        -DBENCH_CELL_IN_MODE=$1 \
        -DBENCH_CELL_OUT_MODE=$2 \
        -DBENCH_CAR_MODE=$3 \
        -DBENCH_SIGNAL_GROUP_MODE=$4 \
        -DBENCH_TRAFFIC_LIGHT_MODE=$5 \
        -DBENCH_LAYOUT_MODE=$6 \
        CMakeLists.txt
  make

  output=$((
  (
    bin/traffic ~/Downloads/graphml_urbanized/Denver--Aurora__CO_Urbanized_Area_23527.graphml
  ) 1>/dev/null
  ) 2>&1)

  echo ${output} >> result_bench.txt
}

for v_layout_mode in "kLayoutModeSoa" "kLayoutModeAos"
do
  # cell_in: [0; 8]
  for i in $(seq -2 8)
  do
    build_and_run ${i} -2 -2 -2 -2 ${v_layout_mode}
  done

  # cell_out: [0; 8]
  for i in $(seq -2 8)
  do
    build_and_run -2 ${i} -2 -2 -2 ${v_layout_mode}
  done

  # car: [0; 30]
  for i in $(seq -2 30)
  do
    build_and_run -2 -2 ${i} -2 -2 ${v_layout_mode}
  done

  # signal group: [0; 8]
  for i in $(seq -2 8)
  do
    build_and_run -2 -2 -2 ${i} -2 ${v_layout_mode}
  done

  # traffic light [0; 6]
  for i in $(seq -2 6)
  do
    build_and_run -2 -2 -2 -2 ${i} ${v_layout_mode}
  done

  build_and_run -1 -1 -1 -1 -1 ${v_layout_mode}
  build_and_run -1 -1 -1 -1 -1 ${v_layout_mode}
  build_and_run 3 3 5 4 4 ${v_layout_mode}
done

