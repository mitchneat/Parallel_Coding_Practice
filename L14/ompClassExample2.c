#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

int main(int argc, char **argv){

  int P = 3;
  omp_set_num_threads(P);

  int N = 100;
  int *a = (int*) calloc(N, sizeof(int));

  int B = (N+P-1)/P;

#pragma omp parallel for
  for(int n=0;n<N;++n){ // fork for loop
    int threadNum = omp_get_thread_num();
    a[n] = threadNum+1;
  } // join

  for(int n=0;n<N;++n){
    printf("a[%d] =%d\n", n, a[n]);
  }

  free(a);
}
