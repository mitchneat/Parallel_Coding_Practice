#include <stdio.h>
#include <stdlib.h>

#include "mpi.h"

int main(int argc, char **argv){

  MPI_Init(&argc, &argv);

  int N = 1;
  int *messageBuffer = (int*) calloc(N, sizeof(int));

  int rank, size;
  
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(rank==0)
    messageBuffer[0] = 999;

  int alive = 1;
  int tag = 123;

  // while the number of ready ranks is less than the total number of ranks
  while(alive<size){

    if(rank<alive){
      int dest = rank+alive;
      if(dest<size){
	// MPI_Send to rank+alive
	MPI_Send(messageBuffer, N, MPI_INT, dest, tag, MPI_COMM_WORLD);
      }
    }
    
    if(rank>=alive && rank<2*alive){
      MPI_Status status;
      int source = rank-alive;
      // MPI_Recv from rank-alive
      MPI_Recv(messageBuffer, N, MPI_INT, source, tag, MPI_COMM_WORLD,
	       &status);
    }
    
    alive *= 2;
  }
  
  printf("rank %d got message %d\n", rank, messageBuffer[0]);

  MPI_Bcast(messageBuffer, N, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Finalize();
  return 0;

}
