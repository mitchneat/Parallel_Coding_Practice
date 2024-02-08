#include<stdlib.h>
#include<stdio.h>
#include<omp.h>


int main(int argc, char **argv){
  int N = 100;
  int *v = (int*) calloc(N, sizeof(int));
  int p = 4;
  omp_set_num_threads(p);

  int b = (N+p-1)/p;

#pragma omp parallel
  {
    int threadIndex = omp_get_thread_num();
    int nstart = b*threadIndex;    
    int len = b*(threadIndex + 1);
    for(int n = nstart; n<len; ++n){
      if (n<N){
	v[n] = threadIndex+1;
      }  
    }
  }

  int n;
  for(n=0; n<N;++n){
    printf("v[%d] = %d\n", n, v[n]);
  }
  free(v);
}
