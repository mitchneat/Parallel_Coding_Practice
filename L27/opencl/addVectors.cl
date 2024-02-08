
__kernel void addVectors(int N, __global float *x, __global float *y, __global float *z){

  int id = get_global_id(0);
  
  if(id<N){
    z[id] = x[id] + y[id];
  }

}
