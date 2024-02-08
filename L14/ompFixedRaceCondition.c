#include <stdlib.h>
#include <stdio.h> 
/* include OpenMP header */
#include <omp.h>

int main(int argc, char **argv){

  int n; /* loop counter */
  
  int N = 100;
  int *v = (int*) calloc(N, sizeof(int));

  omp_set_num_threads(2);

  /* create parallel region with 2 threads */
#pragma omp parallel default(none) private(n) shared(N,v)
  { // fork                                                                          

    int threadIndex = omp_get_thread_num();
    
    // GOOD CODE: n is a private variable
   
    if(threadIndex==0){
      for(n=0;n<N;n+=2){
        v[n] = 1; /* thread 0 sets even entry to 1 */
      }
    }else{
      for(n=1;n<N;n+=2){
        v[n] = 2; /* thread 1 sets odd entry to 2 */
      }
    }
  } // join

  /* serial print out */
  for(n=0;n<N;++n){
    printf("v[%d]=%d\n",n, v[n]);
  }

  return 0;
	     
}
