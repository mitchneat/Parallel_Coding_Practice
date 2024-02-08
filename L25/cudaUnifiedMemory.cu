#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

__global__ void fillKernel(int N, int val, int *c_a){

  int t = threadIdx.x;
  int b = blockIdx.x;
  int B = blockDim.x;

  int n = t + b*B;

  if(n<N)
    c_a[n] = val;

}

int main(int argc, char **argv){

  int N = 1024;
 
  // 1. allocate managed HOST/DEVICE array
  int *u_a;
  cudaMallocManaged(&u_a, N*sizeof(int));

  // 2. launch DEVICE fill kernel   
  int T = 256; // number of threads per thread block
  dim3 G( (N+T-1)/T ); // number of thread blocks to use
  dim3 B(T);

  int val = 999; // value to fill DEVICE array with
  fillKernel <<< G,B >>> (N, val, u_a);

  // 3. synchronize with GPU (block until fillKernel finishes)
  cudaDeviceSynchronize();

  // 4. print out values on HOST    
  for(int n=0;n<N;++n) printf("u_a[%d] = %d\n", n, u_a[n]);

  // 5. free arrays               
  cudaFree(u_a);

  return 0;
}