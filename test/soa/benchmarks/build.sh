set -e
echo "Script will stop on error or incorrect result."

mkdir -p bin
mkdir -p assembly

# Valid addressing mode
g++ -std=c++11 -O0 codegen_test.cc -I../../.. -o bin/valid_gcc_O0
bin/valid_gcc_O0
objdump -S bin/valid_gcc_O0 > assembly/valid_gcc_O0.S

g++ -std=c++11 -O3 codegen_test.cc -I../../.. -o bin/valid_gcc_O3
bin/valid_gcc_O3
objdump -S bin/valid_gcc_O3 > assembly/valid_gcc_O3.S

clang++-3.8 -std=c++11 -O0 codegen_test.cc -I../../.. -o bin/valid_clang38_O0
bin/valid_clang38_O0
objdump -S bin/valid_clang38_O0 > assembly/valid_clang38_O0.S

clang++-3.8 -std=c++11 -O3 codegen_test.cc -I../../.. -o bin/valid_clang38_O3
bin/valid_clang38_O3
objdump -S bin/valid_clang38_O3 > assembly/valid_clang38_O3.S


# Zero addressing mode
g++ -std=c++11 -O0 -DADDR_ZERO codegen_test.cc -I../../.. -o bin/zero_gcc_O0
bin/zero_gcc_O0
objdump -S bin/zero_gcc_O0 > assembly/zero_gcc_O0.S

g++ -std=c++11 -O3 -DADDR_ZERO codegen_test.cc -I../../.. -o bin/zero_gcc_O3
bin/zero_gcc_O3
objdump -S bin/zero_gcc_O3 > assembly/zero_gcc_O3.S

clang++-3.8 -std=c++11 -O0 -DADDR_ZERO codegen_test.cc -I../../.. \
    -o bin/zero_clang38_O0
bin/zero_clang38_O0
objdump -S bin/zero_clang38_O0 > assembly/zero_clang38_O0.S

clang++-3.8 -std=c++11 -O3 -DADDR_ZERO codegen_test.cc -I../../.. \
    -o bin/zero_clang38_O3
bin/zero_clang38_O3
objdump -S bin/zero_clang38_O3 > assembly/zero_clang38_O3.S