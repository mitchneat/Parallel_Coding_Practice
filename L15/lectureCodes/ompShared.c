
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

int main(int argc, char **argv){

  int s = -999, a = 10, b = 5;;

  omp_set_num_threads(4);
  
#pragma omp parallel default(none) shared(s) private(a,b)
  {

    printf("s=%d\n", s);
    s = omp_get_thread_num();

  }
  
  printf("s=%d\n", s);
  return 0;
}
