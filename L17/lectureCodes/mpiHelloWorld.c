
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
  
  printf("Hello world from rank %d of %d\n",
	 rank, size);
  
  MPI_Finalize();

  return 0;

}
