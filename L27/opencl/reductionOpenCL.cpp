#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#ifdef __APPLE__
#include <OpenCL/OpenCl.h>
#else
#include <CL/cl.hpp>
#endif

void pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data)
{
  fprintf(stderr, "OpenCL Error (via pfn_notify): %s\n", errinfo);
}

int main(int argc, char **argv){

  int plat = 0;
  int dev  = 0;

  /* set up CL */
  cl_int            err;
  cl_uint           platforms_n;
  cl_uint           devices_n ;

  cl_context        context;
  cl_command_queue  queue;
  cl_device_id      device;

  /* get number of platforms */
  clGetPlatformIDs(0, NULL, &platforms_n);

  /* get list of platform IDs (platform == implementation of OpenCL) */
  cl_platform_id    *platforms = (cl_platform_id*) calloc(platforms_n, sizeof(cl_platform_id));
  clGetPlatformIDs(platforms_n, platforms, &platforms_n);
  
  if( plat > platforms_n) {
    printf("ERROR: platform %d unavailable \n", plat);
    exit(-1);
  }

  /* get number of devices on platform plat */							  
  cl_uint dtype = CL_DEVICE_TYPE_ALL;

  clGetDeviceIDs(platforms[plat], dtype, 0, NULL, &devices_n);

  // find all available device IDs on chosen platform (could restrict to CPU or GPU)
  cl_device_id *devices = (cl_device_id*) calloc(devices_n, sizeof(cl_device_id));

  clGetDeviceIDs( platforms[plat], dtype, devices_n, devices, &devices_n);
  
  printf("devices_n = %d\n", devices_n);
  
  if(dev>=devices_n){
    printf("invalid device number for this platform\n");
    exit(0);
  }

  // choose user specified device
  device = devices[dev];
  
  // make compute context on device
  context = clCreateContext((cl_context_properties *)NULL, 1, &device, &pfn_notify, (void*)NULL, &err);

  // create command queue
  //queue   = clCreateCommandQueueWithProperties(context, device, NULL, &err);
  queue   = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
  
  // build kernel function
  const char *sourceFileName = "reduction.cl";
  const char *functionName = "reduction";

  // read in text from source file

  struct stat statbuf;
  FILE *fh = fopen(sourceFileName, "r");
  if (fh == 0){
    printf("Failed to open: %s\n", sourceFileName);
    throw 1;
  }
  /* get stats for source file */
  stat(sourceFileName, &statbuf);

  /* read text from source file and add terminator */
  char *source = (char *) malloc(statbuf.st_size + 1);
  fread(source, statbuf.st_size, 1, fh);
  source[statbuf.st_size] = '\0';

  /* create program from source */
  cl_program program = clCreateProgramWithSource(context, 1, (const char **) & source, (size_t*) NULL, &err);

  if (!program){
    printf("Error: Failed to create compute program!\n");
    throw 1;
  }
    
  /* compile and build program */
  const char *allFlags = " ";
  err = clBuildProgram(program, 1, &device, allFlags, (void (*)(cl_program, void*))  NULL, NULL);

  /* check for compilation errors */
  char *build_log;
  size_t ret_val_size;
  err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
  
  build_log = (char*) malloc(ret_val_size+1);
  err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, (size_t*) NULL);
  
  /* to be carefully, terminate with \0
     there's no information in the reference whether the string is 0 terminated or not */
  build_log[ret_val_size] = '\0';

  /* print out compilation log */
  fprintf(stderr, "%s", build_log );

  /* create runnable kernel */
  cl_kernel kernel = clCreateKernel(program, functionName, &err);
  if (! kernel || err != CL_SUCCESS){
    printf("Error: Failed to create compute kernel!\n");
    throw 1;
  }
  
  int N = (argc==2) ? atoi(argv[1]):10240; /* vector size  */

  /* create host array */
  size_t sz = N*sizeof(double);

  double *h_a = (double*) malloc(sz);
  double *h_suma = (double*) malloc(sz);
  for(int n=0;n<N;++n){
    h_a[n] = 1;
  }

  /* create device buffer and copy from host buffer */
  cl_mem c_a    = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sz, h_a, &err);
  cl_mem c_suma = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sz, h_suma, &err);

  /* now set kernel arguments */
  clSetKernelArg(kernel, 0, sizeof(int), &N);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &c_a);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_suma);
  
  /* set thread array */
  int dim = 2;
  size_t Nt = 32;
  size_t Nw = 8;
  size_t B = Nt*Nw;
  size_t Ng = ((N+B-1)/B);
  size_t local[3] = {Nt,Nw,1};
  size_t global[3] = {Ng*Nt,Nw,1};

  /* queue up kernel */
  cl_event event;
  clEnqueueNDRangeKernel(queue, kernel, dim, 0, global, local, 0, (cl_event*)NULL, &event);

  // wait for event
  clWaitForEvents(1, &event);

  /* blocking read from device to host */
  clFinish(queue);
  
  /* blocking read to host */
  sz = Ng*sizeof(double);
  clEnqueueReadBuffer(queue, c_suma, CL_TRUE, 0, sz, h_suma, 0, 0, 0);
  
  /* print out first ten block sums results */
  for(int n=0;n<10;++n)
    printf("h_suma[%d] = %g\n", n, h_suma[n]);

  cl_ulong time_start;
  cl_ulong time_end;
  
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,   sizeof(time_end), &time_end, NULL);
  
  double elapsed = (time_end-time_start)/1.e9;
  printf("elapsed = %e, load throughput = %e GB/s\n", elapsed, (N*sizeof(double)/elapsed)/1e9);


  exit(0);
  
}
