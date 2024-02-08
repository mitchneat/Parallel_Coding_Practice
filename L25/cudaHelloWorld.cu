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
  
  // 1. allocate HOST array
  int *h_a = (int*) calloc(N, sizeof(int));
  int *c_a;
  
  // 2. allocate DEVICE array
  cudaMalloc(&c_a, N*sizeof(int));

  // 3. launch DEVICE fill kernel
  int T = 256;
  dim3 G( (N+T-1)/T );
  dim3 B(T);

  int val = 999;
  fillKernel <<< G,B >>> (N, val, c_a);
  
  // 4. copy data from DEVICE array to HOST array
  cudaMemcpy(h_a, c_a, N*sizeof(int), cudaMemcpyDeviceToHost);
  
  // 5. print out values on HOST
  for(int n=0;n<N;++n){
    printf("h_a[%d] = %d\n", n, h_a[n]);
  }
  
  // 6. free arrays
  cudaFree(c_a);
  free(h_a);
  return 0;
}
