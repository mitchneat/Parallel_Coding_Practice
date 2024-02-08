
#define p_W 32
#define p_NW 8

__kernel void reduction(int N, __global double *c_a, __global double *c_suma){
  
  volatile __local double s_a[p_NW][p_W];
  
  int tx = get_local_id(0);
  int ty = get_local_id(1);
  int b  = get_group_id(0);
  int BX = get_local_size(0);
  int BY = get_local_size(1);
  int Nthreads = p_W*p_NW*get_num_groups(0);
  
  int n = tx + ty*p_W + b*p_W*p_NW;
  double an = 0;
  
  while(n<N){
    an += c_a[n];
    n += Nthreads;
  }
  
  // initial load
  s_a[ty][tx] = an;
  
  // first binary tree reduction (32 to 1 in 8 groups)
  {
    if(tx<16) s_a[ty][tx] += s_a[ty][tx+16];
    if(tx< 8) s_a[ty][tx] += s_a[ty][tx+8];
    if(tx< 4) s_a[ty][tx] += s_a[ty][tx+4];
    if(tx< 2) s_a[ty][tx] += s_a[ty][tx+2];
    if(tx< 1) s_a[ty][ty] = s_a[ty][0] + s_a[ty][1];
  }
  
  // synchronize all warps
  barrier(CLK_LOCAL_MEM_FENCE);

  // second binary tree reduction
  if(ty==0){
    
    // 8 to 1 in one group
    if( tx< 4) s_a[0][tx]  = s_a[tx][tx] + s_a[tx+4][tx+4];
    if( tx< 2) s_a[0][tx] += s_a[0][tx+2];
    if( tx< 1){
      c_suma[b] = s_a[0][0] + s_a[0][1];
    }
  }
}
