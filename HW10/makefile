# set options for this machine
# specify which compilers to use for c, C++, and linking
cc	= mpicc
CC	= mpicc
LD	= mpicc

# define variables
HDRDIR = ./include

# define whether to use the pow function
USE_POW = 0

# optimization
OPTIM = -g

# compiler flags to be used (set to compile with debugging on)
CFLAGS = -I$(HDRDIR) -fopenmp -DUSE_POW=$(USE_POW) $(OPTIM)

# link flags to be used
LDFLAGS	= -fopenmp -DUSE_POW=$(USE_POW) $(OPTIM)

# libraries to be linked in
LIBS	=  -lm

# header files to be counted as dependencies
INCLUDES = include/clusters.h include/data.h

# directory for source files
SRCDIR = ./src

# types of files we are going to construct rules for
.SUFFIXES: .c

# rule for .c files
.c.o: $(INCLUDES) makefile
	$(cc) $(CFLAGS) -o $*.o -c $*.c 

# library objects
OBJS =  src/clusters.o src/data.o

centroids: $(OBJS) $(INCLUDES) makefile src/centroids.o
	$(LD) $(LDFLAGS) -o centroids src/centroids.o $(OBJS) $(LIBS)

kmeans: $(OBJS) $(INCLUDES) makefile src/kmeans.o
	$(LD) $(LDFLAGS) -o kmeans src/kmeans.o $(OBJS) $(LIBS)


# what to do if user types "make clean"
clean:
	rm src/*.o centroids kmeans

