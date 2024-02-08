#include <stdlib.h>
#include <stdio.h> 
/* include OpenMP header */
#include <omp.h>

int main(int argc, char **argv){

  int n; /* loop counter */
  
  int N = 100;
  int *v = (int*) calloc(N, sizeof(int));

  omp_set_num_threads(2);
  
  /* create parallel for loop with 2 threads */
#pragma omp parallel for default(none) private(n) shared(N,v)
  for(n=0;n<N;++n){ /* the omp for pragma splits the iterations between the team of threads executing the parallel region */
    v[n] = n; 
  } // join  


  /* serial print out */
  for(n=0;n<N;++n){
    printf("v[%d]=%d\n",n, v[n]);
  }

  return 0;
	     
}
