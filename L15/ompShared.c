#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


int main(int argc, char **argv){

  int s = -999;
  omp_set_num_threads(4);
  
#pragma omp parallel default(none) private(s)
  {
    printf("s =%d\n", s);
    s = omp_get_thread_num();
    printf("s =%d\n", s);
  }  

  printf("s = %d\n", s);
  
  return 0;
}
