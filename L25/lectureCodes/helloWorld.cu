
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>

// DEVICE CODE
__global__ void helloWorldKernel(){

  int threadIndex = threadIdx.x;
  int blockIndex  = blockIdx.x;
  
  printf("hello world from DEVICE: thread %d in thread-block %d \n",
	 threadIndex, blockIndex);

}

// HOST LAUNCHES DEVICE KERNEL
void helloWorld(){

  // number of thread-blocks
  int G = 10; // 10 blocks of threads

  // number of threads per block
  int B = 32;
  
  helloWorldKernel <<< G, B >>> ();
}

int main(int argc, char **argv){

  printf("Hello world from the HOST\n");

  helloWorld();

  cudaDeviceSynchronize();
  
}
