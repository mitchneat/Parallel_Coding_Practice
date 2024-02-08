    int N = 100000000;
    int *v = (int*) calloc(N, sizeof(int));
    for(int n=0;n<N;++n){
      for(int m=0;m<N;++m){
	v[n] = v[N-1-m];
      }
    }
