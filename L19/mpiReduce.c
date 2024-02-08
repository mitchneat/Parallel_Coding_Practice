#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void myReduce(int msgN, int *msgBuffer, int msgTag);

// to compile:  mpicc -o mpiReduce mpiReduce.c 
int main(int argc, char **argv){
  
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // prepare message
  int msgN = 1;
  int *msgBuffer  = (int*) malloc(msgN*sizeof(int));
  msgBuffer[0] = rank;

  // do sum reduction with MPI_Reduce
  int *msgReduced = (int*) malloc(msgN*sizeof(int));
  int msgRoot = 0;
  MPI_Reduce(msgBuffer, msgReduced, msgN, MPI_INT, MPI_SUM, msgRoot, MPI_COMM_WORLD); 
  

  int msgTag = 999;
  // nominate last process to act as root of tree
  myReduce(msgN, msgBuffer, msgTag);

  if(rank==0) // root process only
    printf("hand rolled reduced value: %d\n", msgBuffer[0]);

  if(rank==msgRoot) // root process only
    printf("MPI_Reduced value: %d\n", msgReduced[0]);

  
  free(msgBuffer);
  
  MPI_Finalize();
  return 0;
}
