# set options for this machine
# specify which compilers to use for c, C++, and linking
cc	= gcc
CC	= g++
LD	= nvcc
NC	= nvcc

# define variables
HDRDIR = ./include

# compiler flags to be used (set to compile with debugging on)
CFLAGS = -I$(HDRDIR) -O3
NFLAGS = -I$(HDRDIR) -O3 --use_fast_math -arch=sm_70

# link flags to be used
LDFLAGS	= --use_fast_math -arch=sm_70

# libraries to be linked in
LIBS	=  -lm

# header files to be counted as dependencies
INCLUDES = include/clusters.h include/data.h

# directory for source files
SRCDIR = ./src

# types of files we are going to construct rules for
.SUFFIXES: .c .cu

# rule for .c files
.c.o: $(INCLUDES)
	$(CC) $(CFLAGS) -o $*.o -c $*.c

# rule for .cu files
.cu.o: $(INCLUDES)
	$(NC) $(NFLAGS) -o $*.o -c $*.cu 

# library objects
OBJS = src/kmeans.o src/clusters.o src/data.o

kmeans: $(OBJS) $(INCLUDES)
	$(LD) $(LDFLAGS) -o kmeans $(OBJS) $(LIBS)

# what to do if user types "make clean"
clean:
	rm src/*.o kmeans

