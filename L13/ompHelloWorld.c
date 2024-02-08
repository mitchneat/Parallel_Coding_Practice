/* L13/ompHelloWorld.c */

#include <stdio.h> 
/* include OpenMP header */
#include <omp.h>

int main(int argc, char **argv){

  /* use OpenMP API function to select 
     number of threads to fork in parallel regions */
  omp_set_num_threads(4);
  
  /* fork a parallel region */
  #pragma omp parallel   
  {
    /* each forked thread will do this */
    printf("Hello world!\n");
  }
  
  return 0;
}

