
#include <stdio.h>
#include <stdlib.h>

#include "mpi.h"

int main(int argc, char **argv){

  int rank, size;

  MPI_Init(&argc, &argv);

  // find out the rank of this process
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  // find out how many processes there are
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  // send message
  // build a message (array is referred to as buffer)
  if(rank==0){ // sender 
    int  sendCount  = 10;
    int *sendBuffer =
      (int*) calloc(sendCount, sizeof(int));
    int  destination = 1;
    int sendTag = 7;
    for(int n=0;n<sendCount;++n){
      sendBuffer[n] = 999;
    }

    MPI_Send(sendBuffer,
	     sendCount,
	     MPI_INT,
	     destination,
	     sendTag,
	     MPI_COMM_WORLD);  
  }

  if(rank==1){ // sender 
    int  recvCount  = 10;
    int *recvBuffer =
      (int*) calloc(recvCount, sizeof(int));
    int  source= 0;
    int recvTag = 7;
    MPI_Status recvStatus;

    MPI_Recv(recvBuffer,
	     recvCount,
	     MPI_INT,
	     source,
	     recvTag,
	     MPI_COMM_WORLD,
	     &recvStatus);

    for(int n=0;n<recvCount;++n){
      printf("recvBuffer[%d]=%d\n",
	     n, recvBuffer[n]);
    }
  }
  
  MPI_Finalize();
  
  return 0;

}
