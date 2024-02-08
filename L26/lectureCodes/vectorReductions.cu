#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

// to compile (on cascades)
// module load cuda
// nvcc -o vectorReductions vectorReductions.cu --arch=sm_70

// number of threads in thread-blocks (kernels V3,V4,V5)
#define p_B 1024


// sum reduction using one thread on GPU
__global__ void reductionKernelV1(int N, double *c_a, double *c_suma){

  c_suma[0] = 0;
  for(int m=0;m<N;++m)
    c_suma[0] += c_a[m];

}

void timeKernelV1(int N, double *c_a, double *c_suma, double *h_suma, double *elapsed){

  // 1. create events 
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  // 2. record start event
  cudaEventRecord(start);

  // 3. launch CUDA DEVICE kernel with one thread in one thread-block
  reductionKernelV1 <<< 1,1 >>> (N, c_a, c_suma);
  
  // 4. insert end record event in stream
  cudaEventRecord(end);
  
  // 8. copy data from DEVICE array to HOST array
  cudaMemcpy(h_suma, c_suma, 1*sizeof(double), cudaMemcpyDeviceToHost);

  // 9. print out elapsed time
  float felapsed;
  cudaEventSynchronize(end);	
  cudaEventElapsedTime(&felapsed, start, end);
  *elapsed = felapsed/1000.; // convert to seconds

}

// one thread with local stack variable for accumulator 
__global__ void reductionKernelV2(int N, double *c_a, double *c_suma){

  double res = 0;

  for(int m=0;m<N;++m)
    res += c_a[m];

  c_suma[0] = res;
}


void timeKernelV2(int N, double *c_a, double *c_suma, double *h_suma, double *elapsed){

  // 1. create events 
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  // 2. record start event
  cudaEventRecord(start);

  // 3. launch CUDA DEVICE kernel with one thread in one thread-block
  reductionKernelV2 <<< 1,1 >>> (N, c_a, c_suma);
  
  // 4. insert end record event in stream
  cudaEventRecord(end);
  
  // 8. copy data from DEVICE array to HOST array
  cudaMemcpy(h_suma, c_suma, 1*sizeof(double), cudaMemcpyDeviceToHost);

  // 9. print out elapsed time
  float felapsed;
  cudaEventSynchronize(end);	
  cudaEventElapsedTime(&felapsed, start, end);
  *elapsed = felapsed/1000.; // convert to seconds

}

// reduction kernel with N threads, but prone to read-write race conflicts
__global__ void reductionKernelV3(int N, double *c_a, double *c_suma){

  int t = threadIdx.x;
  int b = blockIdx.x;
  int B = blockDim.x;

  int n = t + b*B;
  if(n<N){
    // dangerous: will cause read-write race conflicts
    c_suma[0] += c_a[n];
  }
}

void timeKernelV3(int N, double *c_a, double *c_suma, double *h_suma, double *elapsed){

  
  // 1. create events 
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  // 2. calculate number of thread-blocks and threads per thread-block to use
  int T = p_B;
  dim3 G( (N+T-1)/T );
  dim3 B(T);
  
  // 3. record start event
  cudaEventRecord(start);

  // 3.5 zero accumulator
  cudaMemset(c_suma, 0, sizeof(double));
  
  // 4. launch CUDA DEVICE kernel with one thread in one thread-block
  reductionKernelV3 <<< G,B >>> (N, c_a, c_suma);
  
  // 5. insert end record event in stream
  cudaEventRecord(end);
  
  // 6. copy data from DEVICE array to HOST array
  cudaMemcpy(h_suma, c_suma, 1*sizeof(double), cudaMemcpyDeviceToHost);

  // 7. print out elapsed time
  float felapsed;
  cudaEventSynchronize(end);	
  cudaEventElapsedTime(&felapsed, start, end);
  *elapsed = felapsed/1000.; // convert to seconds

}


__global__ void reductionKernelV4(int N, double *c_a, double *c_suma){

  int t = threadIdx.x;
  int b = blockIdx.x;
  int B = blockDim.x;

  int n = t + b*B;
  if(n<N){
    double an = c_a[n];
    // an uninterruptible increment
    atomicAdd(c_suma, an);
  }
}


void timeKernelV4(int N, double *c_a, double *c_suma, double *h_suma, double *elapsed){

  // 1. create events 
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  // 2. calculate number of thread-blocks and threads per thread-block to use
  int T = p_B;
  dim3 G( (N+T-1)/T );
  dim3 B(T);
  
  // 3. record start event
  cudaEventRecord(start);

  // 3.5 zero accumulator
  cudaMemset(c_suma, 0, sizeof(double));
  
  // 4. launch CUDA DEVICE kernel with one thread in one thread-block
  reductionKernelV4 <<< G,B >>> (N, c_a, c_suma);
  
  // 5. insert end record event in stream
  cudaEventRecord(end);
  
  // 6. copy data from DEVICE array to HOST array
  cudaMemcpy(h_suma, c_suma, 1*sizeof(double), cudaMemcpyDeviceToHost);

  // 7. print out elapsed time
  float felapsed;
  cudaEventSynchronize(end);	
  cudaEventElapsedTime(&felapsed, start, end);
  *elapsed = felapsed/1000.; // convert to seconds

}

// shared memory
__global__ void reductionKernelV5(int N, double *c_a, double *c_suma){

  __shared__ double s_a[p_B];
  
  int t = threadIdx.x;
  int b = blockIdx.x;
  int B = blockDim.x;

  int n = t + b*B;
  double an = (n<N) ? c_a[n]:0;

  s_a[t] = an;

  int alive = p_B/2;
  while(alive>0){

    __syncthreads();
    
    if(t<alive)
      s_a[t] += s_a[t+alive];
    alive /= 2;
  }

  if(t==0){
    double an = s_a[0];
    // an uninterruptible increment
    atomicAdd(c_suma, an);
  }
}

void timeKernelV5(int N, double *c_a, double *c_suma, double *h_suma, double *elapsed){

  // 1. create events 
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  // 2. calculate number of thread-blocks and threads per thread-block to use
  int T = p_B;
  dim3 G( (N+T-1)/T );
  dim3 B(T);
  
  // 3. record start event
  cudaEventRecord(start);

  // 3.5 zero accumulator
  cudaMemset(c_suma, 0, sizeof(double));
  
  // 4. launch CUDA DEVICE kernel with one thread in one thread-block
  reductionKernelV5 <<< G,B >>> (N, c_a, c_suma);
  
  // 5. insert end record event in stream
  cudaEventRecord(end);
  
  // 6. copy data from DEVICE array to HOST array
  cudaMemcpy(h_suma, c_suma, 1*sizeof(double), cudaMemcpyDeviceToHost);

  // 7. print out elapsed time
  float felapsed;
  cudaEventSynchronize(end);	
  cudaEventElapsedTime(&felapsed, start, end);
  *elapsed = felapsed/1000.; // convert to seconds

}

// use warp synchrony
// (32 threads in each SIMD group are synchronized via syncwarp)
#define p_W 32

__global__ void reductionKernelV6(int N, double *c_a, double *c_suma){

  __shared__ double s_a[p_W][p_W];
  
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int b  = blockIdx.x;
  int BX = blockDim.x;
  int BY = blockDim.y;

  int n = tx + ty*BX + b*BX*BY;
  double an = (n<N) ? c_a[n]:0;

  // initial load
  s_a[ty][tx] = an;
  __syncwarp();

  // first binary tree reduction
  {
    if(tx<16) s_a[ty][tx] += s_a[ty][tx+16];
    __syncwarp();
    
    if(tx< 8) s_a[ty][tx] += s_a[ty][tx+8];
    __syncwarp();
    
    if(tx< 4) s_a[ty][tx] += s_a[ty][tx+4];
    __syncwarp();
    
    if(tx< 2) s_a[ty][tx] += s_a[ty][tx+2];
    __syncwarp();
    
    if(tx< 1) s_a[ty][ty] = s_a[ty][0] + s_a[ty][1];
  }

  // synchronize all warps
  __syncthreads();

  // second binary tree reduction
  if(ty==0){
    if( tx<16) s_a[0][tx] = s_a[tx][tx] + s_a[tx+16][tx+16];
    __syncwarp();
    
    if( tx< 8) s_a[0][tx] += s_a[0][tx+8];
    __syncwarp();
    
    if( tx< 4) s_a[0][tx] += s_a[0][tx+4];
    __syncwarp();
    
    if( tx< 2) s_a[0][tx] += s_a[0][tx+2];
    __syncwarp();
    
    if( tx< 1){
      double res = s_a[0][0] + s_a[0][1];
      // an uninterruptible increment
      atomicAdd(c_suma, res);
    }
  }
}

void timeKernelV6(int N, double *c_a, double *c_suma, double *h_suma, double *elapsed){

  // 1. create events 
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  // 2. calculate number of thread-blocks and threads per thread-block to use
  //    using 32 x 32 threads per thread block
  int W = p_W;
  dim3 G2D( (N+W*W-1)/(W*W) );
  dim3 B2D(W,W);
  
  // 3. record start event
  cudaEventRecord(start);

  // 3.5 zero accumulator
  cudaMemset(c_suma, 0, sizeof(double));
  
  // 4. launch CUDA DEVICE kernel with one thread in one thread-block
  reductionKernelV6 <<< G2D,B2D >>> (N, c_a, c_suma);
  
  // 5. insert end record event in stream
  cudaEventRecord(end);
  
  // 6. copy data from DEVICE array to HOST array
  cudaMemcpy(h_suma, c_suma, 1*sizeof(double), cudaMemcpyDeviceToHost);

  // 7. print out elapsed time
  float felapsed;
  cudaEventSynchronize(end);	
  cudaEventElapsedTime(&felapsed, start, end);
  *elapsed = felapsed/1000.; // convert to seconds

}


// reduce number of tree reductions - each thread loads multiple values
__global__ void reductionKernelV7(int N, double *c_a, double *c_suma){

  __shared__ double s_a[p_W][p_W];
  
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int b  = blockIdx.x;
  int BX = blockDim.x;
  int BY = blockDim.y;
  int Nthreads = BX*BY*gridDim.x;
  
  int n = tx + ty*BX + b*BX*BY;
  double an = 0;

  while(n<N){
    an += c_a[n];
    n += Nthreads;
  }

  // initial load
  s_a[ty][tx] = an;
  __syncwarp();

  // first binary tree reduction
  {
    if(tx<16) s_a[ty][tx] += s_a[ty][tx+16];
    __syncwarp();
    
    if(tx< 8) s_a[ty][tx] += s_a[ty][tx+8];
    __syncwarp();
    
    if(tx< 4) s_a[ty][tx] += s_a[ty][tx+4];
    __syncwarp();
    
    if(tx< 2) s_a[ty][tx] += s_a[ty][tx+2];
    __syncwarp();
    
    if(tx< 1) s_a[ty][ty] = s_a[ty][0] + s_a[ty][1];
  }

  // synchronize all warps
  __syncthreads();

  // second binary tree reduction
  if(ty==0){
    if( tx<16) s_a[0][tx] = s_a[tx][tx] + s_a[tx+16][tx+16];
    __syncwarp();
    
    if( tx< 8) s_a[0][tx] += s_a[0][tx+8];
    __syncwarp();
    
    if( tx< 4) s_a[0][tx] += s_a[0][tx+4];
    __syncwarp();
    
    if( tx< 2) s_a[0][tx] += s_a[0][tx+2];
    __syncwarp();
    
    if( tx< 1){
      double res = s_a[0][0] + s_a[0][1];
      // an uninterruptible increment
      atomicAdd(c_suma, res);
    }
  }
}

void timeKernelV7(int N, double *c_a, double *c_suma, double *h_suma, double *elapsed){

  // 1. create events 
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  // 2. calculate number of thread-blocks and threads per thread-block to use
  //    using 32 x 32 threads per thread block
  int W = p_W;
  int Nthreads = N/4; // reduce number of thread blocks by factor of 4
  dim3 G2D( (Nthreads+W*W-1)/(W*W) );
  dim3 B2D(W,W);
  
  // 3. record start event
  cudaEventRecord(start);

  // 3.5 zero accumulator
  cudaMemset(c_suma, 0, sizeof(double));
  
  // 4. launch CUDA DEVICE kernel with one thread in one thread-block
  reductionKernelV7 <<< G2D,B2D >>> (N, c_a, c_suma);
  
  // 5. insert end record event in stream
  cudaEventRecord(end);
  
  // 6. copy data from DEVICE array to HOST array
  cudaMemcpy(h_suma, c_suma, 1*sizeof(double), cudaMemcpyDeviceToHost);

  // 7. print out elapsed time
  float felapsed;
  cudaEventSynchronize(end);	
  cudaEventElapsedTime(&felapsed, start, end);
  *elapsed = felapsed/1000.; // convert to seconds

}


int main(int argc, char **argv){

  int N = (argc==2) ? atoi(argv[1]):1000;

  // 0. number of kernels to test
  int Nkernels = 7;
  
  // 1. allocate HOST array
  double *h_a = (double*) calloc(N, sizeof(double));
  double *h_suma = (double*) calloc(Nkernels, sizeof(double));
  double *c_a, *c_suma;

  for(int n=0;n<N;++n){
    h_a[n] = 1;
  }
  
  // 2. allocate DEVICE array
  cudaMalloc(&c_a, N*sizeof(double));
  cudaMalloc(&c_suma, 1*sizeof(double));
  
  cudaMemcpy(c_a, h_a, N*sizeof(double), cudaMemcpyHostToDevice);

  // 3. time kernels
  double *elapsedTimes = (double*) calloc(Nkernels, sizeof(double));
  timeKernelV1(N, c_a, c_suma, h_suma+0, elapsedTimes+0);
  timeKernelV2(N, c_a, c_suma, h_suma+1, elapsedTimes+1);
  timeKernelV3(N, c_a, c_suma, h_suma+2, elapsedTimes+2);
  timeKernelV4(N, c_a, c_suma, h_suma+3, elapsedTimes+3);
  timeKernelV5(N, c_a, c_suma, h_suma+4, elapsedTimes+4);
  timeKernelV6(N, c_a, c_suma, h_suma+5, elapsedTimes+5);
  timeKernelV7(N, c_a, c_suma, h_suma+6, elapsedTimes+6);

  // 4. print results
  printf("Kernel, elapsed, data load (GB/s), result\n");
  for(int knl=0;knl<Nkernels;++knl){
    printf("%d &  % 3.2e &  % 3.2e &  % d \\\\ \n",
	   knl+1, elapsedTimes[knl],
	   sizeof(double)*N/(elapsedTimes[knl]*1.e9), (int)h_suma[knl]);
  }
  
  // 5. free arrays
  cudaFree(c_a);
  free(h_a);
  return 0;
}
