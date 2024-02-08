#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int modulo(int x,int N){
  return ((x%N) + N) %N;
}


void myBroadcast(int msgN, int *msgOut, int *msgIn, int msgTag){

  // find process rank and process count
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int alive = 1;
  while(alive<size){ // loop over log2(size) rounds
    int newAlive  = 2*alive;
    int msgDest   = rank + alive;
    int msgSource = rank - alive;
    
    if(rank<alive && msgDest<newAlive && msgDest<size){ // send
      MPI_Send(msgOut, msgN, MPI_INT, msgDest, msgTag, MPI_COMM_WORLD);
    }
    
    if(rank<newAlive && msgSource>=0){
      MPI_Status msgStatus;
      MPI_Recv(msgIn, msgN, MPI_INT, msgSource, msgTag, MPI_COMM_WORLD, &msgStatus);
    }
    alive = newAlive;
  }
}

void myBroadcastFromRoot(int msgN, int msgRoot, int *msgOut, int *msgIn, int msgTag){

  // find process rank and process count
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  int alive = 1;
  int mapRank = modulo(rank-msgRoot,size);
  
  while(alive<size){ // loop over log2(size) rounds
    int newAlive  = 2*alive;
    int msgDest   = modulo(rank + alive, size);
    int msgSource = modulo(rank - alive, size);
    
    if(mapRank<alive && (mapRank+alive)<newAlive && (mapRank+alive)<size){ // send
      MPI_Send(msgOut, msgN, MPI_INT, msgDest, msgTag, MPI_COMM_WORLD);
    }
    
    if(mapRank>=alive && mapRank<newAlive){
      MPI_Status msgStatus;

      MPI_Recv(msgIn, msgN, MPI_INT, msgSource, msgTag, MPI_COMM_WORLD, &msgStatus);
    }
    alive = newAlive;
  }

  
}
  
// to compile:  mpicc -o mpiBroadcast mpiBroadcast.c
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

  myBroadcast(msgN, msgOut, msgIn, msgTag);

  // inject MPI barrier into timeline
  MPI_Barrier(MPI_COMM_WORLD);
  
  // nominate last process to act as root of tree
  int msgRoot = size-1;
  myBroadcastFromRoot(msgN, msgRoot, msgOut, msgIn, msgTag);

  // nominate first process as root
  msgRoot = 0;
  myBroadcastFromRoot(msgN, msgRoot, msgOut, msgIn, msgTag);
  
  MPI_Finalize();
  return 0;
}
