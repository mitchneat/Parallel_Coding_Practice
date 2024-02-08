#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// to compile:  mpicc -o mpiTreeBarrier mpiTreeBarrier.c
int main(int argc, char **argv){
  
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  int msgN = 1;
  int msgTag = 999;
  
  int *msgOut = (int*) malloc(msgN*sizeof(int));
  int *msgIn  = (int*) malloc(msgN*sizeof(int));
  
  msgOut[0] = rank;
  
  int alive = size;
  
  while(alive>1){ // loop over log2(size) rounds
    int halfAlive = (alive+1)/2;
    if(rank<alive && rank>=halfAlive){ // send
      int msgDest = rank - halfAlive;
      MPI_Send(msgOut, msgN, MPI_INT, msgDest, msgTag, MPI_COMM_WORLD);
    }

    if(rank<halfAlive){
      MPI_Status msgStatus;
      int msgSource = rank + halfAlive;
      if(msgSource<size && msgSource<alive){ // receive
	MPI_Recv(msgIn, msgN, MPI_INT, msgSource, msgTag, MPI_COMM_WORLD, &msgStatus);
      }
    }
    alive = halfAlive;
  }

  MPI_Finalize();
  return 0;
}
