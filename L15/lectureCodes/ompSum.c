#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

double sumSerial(int N, double *a){

  double s = 0;
  int n;
  
  for(n=0;n<N;++n)
    s = s + a[n];

  return s;
}

double sumOmpNaive(int N, double *a){

  double s = 0;
  int n;

#pragma omp parallel for
  for(n=0;n<N;++n)
    s = s + a[n];

  return s;
}


double sumOmpCritical(int N, double *a){

  double s = 0;
  int n;

#pragma omp parallel for
  for(n=0;n<N;++n){

#pragma omp critical
    {
      s = s + a[n];
    }
    
  }
  
  return s;
}


double sumOmpAtomic(int N, double *a){

  double s = 0;
  int n;

#pragma omp parallel for
  for(n=0;n<N;++n){

#pragma omp atomic
    s = s + a[n];
  }
  
  return s;
}

double sumOmpReduction(int N, double *a){

  double s = 0;
  int n;

#pragma omp parallel for reduction(+:s)
  for(n=0;n<N;++n){
    s = s + a[n];
  }
  
  return s;
}


int main(int argc, char **argv){

  double s;
  int N = atoi(argv[1]);

  double *a = (double*) calloc(N, sizeof(double));
  for(int n=0;n<N;++n)
    a[n] = 1;

  omp_set_num_threads(4);

  double t0 = omp_get_wtime();  
  double s0 = sumSerial(N, a);

  double t1 = omp_get_wtime();
  double s1 = sumOmpNaive(N, a);

  double t2 = omp_get_wtime();
  double s2 = sumOmpCritical(N, a);

  double t3 = omp_get_wtime();
  double s3 = sumOmpAtomic(N, a);

  double t4 = omp_get_wtime();
  double s4 = sumOmpReduction(N, a);

  double t5 = omp_get_wtime();

  printf("sumSerial:       %f took %e\n", s0, t1-t0);
  printf("sumOmpNaive:     %f took %e\n", s1, t2-t1);
  printf("sumOmpCritical:  %f took %e\n", s2, t3-t2);
  printf("sumOmpAtomic:    %f took %e\n", s3, t4-t3);
  printf("sumOmpReduction: %f took %e\n", s4, t5-t4);
  
}
