
#include <stdio.h>
#include <stdlib.h>

#include "mpi.h"

int main(int argc, char **argv){

  MPI_Init(&argc, &argv);

  int rank, size;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  int N = 10;
  int *sendBuffer = (int*) calloc(N, sizeof(int)); 
  int *recvBuffer = (int*) calloc(N, sizeof(int));

  for(int n=0;n<N;++n){
    sendBuffer[n] = rank;
  }

  int tag = 999;
  int source = (rank>0) ? rank-1: size-1;
  MPI_Status status;

  MPI_Request sendRequest, recvRequest;
  
  MPI_Irecv(recvBuffer, N, MPI_INT, source, tag, 
	    MPI_COMM_WORLD, &recvRequest);

  int dest = (rank<size-1) ? rank+1: 0;
  MPI_Isend(sendBuffer, N, MPI_INT, dest, tag, 
	    MPI_COMM_WORLD, &sendRequest);

  for(int i=0;i<1000000;++i){
    do stuff;
  }

  // block until data is received
  MPI_Wait(&recvRequest, &status);
  MPI_Wait(&sendRequest, &status);
  
  printf("message received by rank %d is %d\n",
	 rank, recvBuffer[0]);


  
  MPI_Finalize();

}
