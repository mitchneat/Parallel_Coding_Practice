#include <omp.h>

// #1. In your makefile make sure that -fopenmp shows up in CFLAGS and LDFLAGS

// #2. Avoid making the innermost loop in a nested loop parallel

// #3. Some comments on thread safe functions:

// ok to call from inside an openmp loop
int goodFunction(int a, int b){
  int d = a+b; // private variable
  return d;
}

// bad - do not call from openmp loop (c is shared)

int c; // shared variable

int badFunction1(int a, int b){

  c+=a;

  return c+b;
}

// bad - do not call from openmp function - d is static and thus shared 
int badFunction2(int a, int b){
  static int d;
  d += a;

  return d+b;
}

// maybe don't use - if all threads touch different entries in b this is ok 
void possiblyBadFunction(int a, int *b){

  // it depends on what values of a are passed by all threads
  b[a] += 1;
  
}
