#include <stdio.h>
#include <stdlib.h>

#include "mpi.h"

double mpiStartTimer(){

  // make sure all MPI processes have completed prior tasks
  MPI_Barrier(MPI_COMM_WORLD);
  
  // get start time
  return MPI_Wtime();
}

double mpiEndTimer(double start){
  
  // get wall clock end time
  double end = MPI_Wtime();
  
  // find maximum elapsed time on any process
  int N = 1;
  double *message = (double*) calloc(N, sizeof(double));
  double *elapsed = (double*) calloc(N, sizeof(double));
  message[0] = end-start;
  MPI_Allreduce(message, elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  
  double res = elapsed[0];
  free(elapsed);
  free(message);
  
  return res;
}

int main(int argc, char **argv){

  MPI_Init(&argc, &argv);
  
  // find rank
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // number of entries in local array
  int Nlocal = atoi(argv[1]);
  double *v = (double*) calloc(Nlocal, sizeof(double));

  for(int n=0;n<Nlocal;++n)
    v[n] = 1;
  
  // set up message buffer for reduction
  int N = 1;
  double *message = (double*) calloc(N, sizeof(double));
  double *answer  = (double*) calloc(N, sizeof(double));

  // start timer =====================================>
  double start = mpiStartTimer();

  // locally reduce array
  double res = 0;
  for(int n=0;n<Nlocal;++n)
    res += v[n];
  message[0] = res;
  
  // perform distributed sum reduction
  int root = 0; // destination for sum of distributed message entries
  MPI_Reduce(message, answer, N, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

  // compute elapsed time <==============================
  double elapsed = mpiEndTimer(start);

  
  if(rank==root){
    if(size==1) printf(" %%%% MPI size, elapsed\n");
    printf("(%d, %g)\n", size, elapsed);
  }

  MPI_Finalize();
  return 0;
}
