flags = -m64 -O3 

ifeq ($(shell uname -s),Darwin)
libs = -framework OpenCL
else
libs = -lOpenCL
endif


addVectors: addVectorsOpenCL.cpp
	g++ -o addVectorsOpenCL $(flags) addVectorsOpenCL.cpp $(libs)

reduction: reductionOpenCL.cpp
	g++ -o reductionOpenCL $(flags) reductionOpenCL.cpp $(libs)

clean:
	rm -f addVectorsOpenCL.o addVectorsOpenCL reductionOpenCL reductionOpenCL.o
