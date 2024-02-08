#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


/* To compile: 
   gcc -fopenmp -o ompSum ompSum.c -lm 
   
   To run with 10000 values:
   ./ompSum 10000 
*/

/* all threads update shared variable */
float sumParallelForShared(int N, float *a){
  /* BAD CODE - causes race condition */
  float s = 0;

#pragma omp parallel for shared(s)
  for(int n=0;n<N;++n){
    s = s+a[n];
  }

  printf("sumParallelFor          computes sum %f\n", s);
  
  return s;
}

float sumParallelForCritical(int N, float *a){
  /* CORRECT CODE - slowed by critical condition */
  float s = 0;

#pragma omp parallel for shared(s)
  for(int n=0;n<N;++n){
#pragma omp critical
    s = s+a[n];
  }

  printf("sumParallelForCritical  computes sum %f\n", s);
  
  return s;
}

float sumParallelForAtomic(int N, float *a){
  /* GOOD CODE - slowed by atomics */
  float s = 0;

#pragma omp parallel for shared(s)
  for(int n=0;n<N;++n){
#pragma omp atomic
    s = s+a[n];
  }

  printf("sumParallelForAtomic    computes sum %f\n", s);
  
  return s;
}


double sumParallelForReduction(int N, double *a){
  /* GOOD CODE */
  double s = 0;

#pragma omp parallel for reduction(+:s)
  for(int n=0;n<N;++n){
    s = s+a[n];
  }

  printf("sumParallelForReduction computes sum %f\n", s);
  
  return s;
}


int main(int argc, char **argv){

  // read number of entries from command line
  int N = (argc==2) ? atoi(argv[1]): 1000;
  
  double *a = (double*) calloc(N, sizeof(double));
  double s;
  double t0, t1, t2, t3, t4;

  // populate array
  for(int n=0;n<N;++n)
    a[n] = 1;

  t0 = omp_get_wtime();
  
  // sum using parallel for and shared accumulator
  sumParallelForShared(N, a);

  t1 = omp_get_wtime();
  
  // sum using critical region
  s = sumParallelForCritical(N, a);
  
  t2 = omp_get_wtime();

  // sum using atomic operation
  s = sumParallelForAtomic(N, a);

  t3 = omp_get_wtime();
  
  // sum using reduction clause
  s = sumParallelForReduction(N, a);

  t4 = omp_get_wtime();

  // output timing intervals
  printf("parallel for shared    took: %f\n", t1-t0);
  printf("parallel for critical  took: %f\n", t2-t1);
  printf("parallel for atomic    took: %f\n", t3-t2);
  printf("parallel for reduction took: %f\n", t4-t3);
  
  return 0;
}
