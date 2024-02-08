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
    int threadNumber = omp_get_thread_num();
  
    if(threadNumber%2 == 0) /* compute threadNumber mod 2 */
      printf("Hello world!\n"); /* each even thread does this */
    else
      printf("Goodbye world!\n"); /* each odd thread does this */
  }
  
  return 0;
}
