#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void myReduce(int msgN, int *msgBuffer, int msgTag){

  // find process rank and process count
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int *msgTmp = (int*) calloc(msgN, sizeof(int));
  
  int alive = size;
  while(alive>1){ 
    int newAlive  = (alive+1)/2;
    int msgDest   = rank - newAlive;
    int msgSource = rank + newAlive;

    // recv from above
    if(rank<newAlive && msgSource>=newAlive && msgSource<alive){ // send
      MPI_Status msgStatus;
      MPI_Recv(msgTmp, msgN, MPI_INT, msgSource, msgTag, MPI_COMM_WORLD, &msgStatus);

      // add incoming values to outgoing buffer
      for(int n=0;n<msgN;++n){
	msgBuffer[n] += msgTmp[n];
      }
    }
    
    // send down
    if(rank>=newAlive && rank<alive && msgDest<newAlive){
      MPI_Send(msgBuffer, msgN, MPI_INT, msgDest, msgTag, MPI_COMM_WORLD);
    }

    // update alive
    alive = newAlive;
  }

  free(msgTmp);
  
}
