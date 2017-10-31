// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA with an array
// of offsets. Then the offsets are added in parallel to produce the string "World!"
// By Ingemar Ragnemalm 2010
 
#include <stdio.h>
//#include "executor/cuda_executor.h"
#include "soa/soa.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

using ikra::soa::IndexType;
using ikra::soa::SoaLayout;
using ikra::soa::kAddressModeZero;
using ikra::soa::DynamicStorage;

__device__ char data_buffer[10000];



class Vertex : public SoaLayout<Vertex, 1000> {
 public:
  IKRA_INITIALIZE_CLASS(data_buffer)

  __device__ Vertex(int a) {
    printf("IN CONSTRUCTOR!!\n");
  }


/*
  int_ field0;
  int_ field1;

  __ikra_device__ void add_fields() {
    field0 = field0 + field1;
  }
*/

};


template<typename T>
__global__ void myKernel() {
  new (Vertex::get_(0)) Vertex(123);
}



int main()
{
  void* bla;
  cudaMalloc( (void**)&bla, 100 );
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  
  myKernel<int><<<1,2>>>();
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  printf("!!!!!\n");
}


// Keep nullptr as special "not an object"