#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

int main(int argc, char **argv){

  int P = 3;
  omp_set_num_threads(P);

  int N = 6;
  int *a = (int*) calloc(N, sizeof(int));

  int B = (N+P-1)/P;

  a[0] = 0;
  
  // BAAAAAD
#pragma omp parallel for
  for(int n=1;n<N;++n){ // fork for loop
    a[n] = a[n-1]+1;
  } // join

  for(int n=0;n<N;++n){
    printf("a[%d] =%d\n", n, a[n]);
  }

  free(a);
}
