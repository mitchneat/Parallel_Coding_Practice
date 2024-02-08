
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>

// DEVICE CODE
__global__ void addArraysKernel(int N, float *c_a, float *c_b, float *c_c){

  int threadIndex = threadIdx.x;
  int blockIndex  = blockIdx.x;
  int B           = blockDim.x;
  
  int n = (threadIndex + B*blockIndex);

  if(n<N)
    c_c[n] = c_a[n] + c_b[n];
  
}

// HOST LAUNCHES DEVICE KERNEL
void addArrays(int N, float *c_a, float *c_b, float *c_c){

  // number of threads per block
  int B = 32;

  // number of thread-blocks
  int G = (N+B-1)/B; 
  
  addArraysKernel <<< G, B >>> (N, c_a, c_b, c_c);
}

int main(int argc, char **argv){

  int N = atoi(argv[1]);

  float *h_a = (float*) calloc(N, sizeof(float));
  float *h_b = (float*) calloc(N, sizeof(float));
  float *h_c = (float*) calloc(N, sizeof(float));

  for(int n=0;n<N;++n){
    h_a[n] = 1-n;
    h_b[n] = 1+n;
  }
  
  float *c_a, *c_b, *c_c;

  // allocate arrays on DEVICE
  cudaMalloc(&c_a, N*sizeof(float));
  cudaMalloc(&c_b, N*sizeof(float));
  cudaMalloc(&c_c, N*sizeof(float));

  // copy data from HOST to DEVICE
  cudaMemcpy(c_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(c_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice);
  
  addArrays(N, c_a, c_b, c_c);

  // copy data result from DEVICE to HOST
  cudaMemcpy(h_c, c_c, N*sizeof(float), cudaMemcpyDeviceToHost);

  // print out result
  for(int n=0;n<N;++n)
    printf("h_c[%d] = %e\n", n, h_c[n]);
}
