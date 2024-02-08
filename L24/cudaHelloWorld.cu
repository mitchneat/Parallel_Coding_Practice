
#include <stdio.h>
#include <stdlib.h>

#include "cuda.h"

/* 

#1. To get time on a cascades compute node with an NVIDIA V100 GPU:

salloc --partition=v100_dev_q --nodes=1  --gres=gpu:1 -Acmda3634

#2. To load the CUDA development kit when on the interactive compute node:

module load cuda/10.1.168

#3. To compile this program

nvcc -o cudaHelloWorld cudaHelloWorld.cu

#4. To run

./cudaHelloWorld

*/


__global__ void helloWorldKernel (){

  printf("hello world from the gpu\n");

}

int main(int argc, char **argv){

    printf("hello world from CPU\n");

    helloWorldKernel <<< 80, 32 >>> ();

    // barrier until all GPU tasks have completed
    cudaDeviceSynchronize();

    return 0;

}
