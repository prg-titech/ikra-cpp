// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA with an array
// of offsets. Then the offsets are added in parallel to produce the string "World!"
// By Ingemar Ragnemalm 2010
 
#include <stdio.h>
#include "executor/executor.h"
#include "soa/soa.h"

const int N = 16; 
const int blocksize = 16; 

using ikra::soa::IndexType;
using ikra::soa::SoaLayout;
using ikra::soa::kAddressModeZero;
using ikra::soa::DynamicStorage;
using ikra::executor::execute;
using ikra::executor::execute_and_reduce;

char data_buffer[10000];

class Vertex : public SoaLayout<Vertex, 1000> {
 public:
  IKRA_INITIALIZE_CLASS(data_buffer)

  int_ field0;
  int_ field1;
};

__global__ 
void hello(char *a, int *b) 
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  Vertex::get(tid)->field0 += Vertex::get(tid)->field1;
}
 
int main()
{
  char a[N] = "Hello \0\0\0\0\0\0";
  int b[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
 
  char *ad;
  int *bd;
  const int csize = N*sizeof(char);
  const int isize = N*sizeof(int);
 
  printf("%s", a);
 
  cudaMalloc( (void**)&ad, csize ); 
  cudaMalloc( (void**)&bd, isize ); 
  cudaMemcpy( ad, a, csize, cudaMemcpyHostToDevice ); 
  cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice ); 
  
  dim3 dimBlock( blocksize, 1 );
  dim3 dimGrid( 1, 1 );
  hello<<<dimGrid, dimBlock>>>(ad, bd);
  cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost ); 
  cudaFree( ad );
  cudaFree( bd );
  
  printf("%s\n", a);
  return EXIT_SUCCESS;
}
