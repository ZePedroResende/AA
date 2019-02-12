#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "papi.h"
#include <sys/time.h>
#define BLOCK_SIZE 32
#define TIME_RESOLUTION 1000000// time measuring resolution (us)

double clearcache[30000000];
long long unsigned initial_time;
timeval t;
int NUM_EVENTS=0;
int *PAPI_events;
long long *counters;

void clearCache (void) {
		  for (unsigned i = 0; i < 30000000; ++i)
				      clearcache[i] = i;
}

void printResults (long long unsigned tt) {
		  printf("%llu", tt);
}

void start (void) {
		  gettimeofday(&t, NULL);
		    initial_time = t.tv_sec * TIME_RESOLUTION + t.tv_usec;
}

long long unsigned stop (void) {
		  gettimeofday(&t, NULL);
		    long long unsigned final_time = t.tv_sec * TIME_RESOLUTION + t.tv_usec;

			  return final_time - initial_time;
}

void validate_rows(float *C, int SIZE){
		int i, j, r = 1;

		for (i = 0; i < SIZE && r; i++){
				for (j = 0; j < SIZE && r; j++){
						if (C[i * SIZE + j] != C[i * SIZE + 0]) r = 0;
				}
		}

		if(!r) {
				printf("ERRRO");
		}
}

void validate_columns(float *C, int SIZE){
		int i, j, r = 1;

		for (i = 0; i < SIZE && r; i++){
				for (j = 0; j < SIZE && r; j++){
						if (C[i * SIZE + j] != C[0 * SIZE + j]) r = 0;
				}
		}

		if(!r) {
				printf("ERRRO");
		}
}

void fillMatrices (float *A,float *B,float *C, int SIZE) {
  for (int i = 0; i < SIZE; ++i) {
    for (int j = 0; j < SIZE; ++j) {
      A[i * SIZE + j] = ((float) rand()) / ((float) RAND_MAX);
      B[i * SIZE + j] = 1;
      C[i * SIZE + j] = 0;
    }
  }
}


__global__ void testKernel (float *Da, float *Db, float *Dc, int N) {
}

void comCUDA (float*a, float*b, float*c, int N) {
  float *Da, *Db, *Dc;
  cudaMalloc( (void**) &Da, N * N *sizeof(float) );
  cudaMalloc( (void**) &Db, N * N *sizeof(float) );
  cudaMalloc( (void**) &Dc, N * N *sizeof(float) );

  cudaMemcpy(Da, a, N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(Db, b, N * N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimBlock (BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid (N/BLOCK_SIZE+(N%BLOCK_SIZE>0), N/BLOCK_SIZE+(N%BLOCK_SIZE>0) );
  testKernel<<<dimGrid, dimBlock>>>(Da, Db, Dc, N);

  cudaDeviceSynchronize();
  cudaMemcpy(c, Dc, N * N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(Da);
  cudaFree(Db);
  cudaFree(Dc);
}


__global__ void cudaKernel (float *Da, float *Db, float *Dc, int N) {
  int COL = threadIdx.x + blockDim.x * blockIdx.x;
  int ROW = threadIdx.y + blockDim.y * blockIdx.y;
  int k;

  if (ROW>=N || COL>=N){
    float res=0.0f;
    for(k=0; k<N; ++k){
      res += Da[ROW*N + k] * Db[k*N + COL];
    }
    Dc[ROW*N + COL]=res;
  }
}

void dotProductCUDA (float*a, float*b, float*c, int N) {
  float *Da, *Db, *Dc;
  cudaMalloc( (void**) &Da, N * N *sizeof(float) );
  cudaMalloc( (void**) &Db, N * N *sizeof(float) );
  cudaMalloc( (void**) &Dc, N * N *sizeof(float) );

  cudaMemcpy(Da, a, N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(Db, b, N * N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimBlock (BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid (N/BLOCK_SIZE+(N%BLOCK_SIZE>0), N/BLOCK_SIZE+(N%BLOCK_SIZE>0) );
  cudaKernel<<<dimGrid, dimBlock>>>(Da, Db, Dc, N);

  cudaDeviceSynchronize();
  cudaMemcpy(c, Dc, N * N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(Da);
  cudaFree(Db);
  cudaFree(Dc);
}

__global__ void cudaBlockKernel(float *a, float *b, float *c, int N){
  float CValue = 0;

  int Row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
  int Col = blockIdx.x*BLOCK_SIZE + threadIdx.x;

  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

  for (int k = 0; k < (BLOCK_SIZE + N - 1)/BLOCK_SIZE; k++) {

    if (k*BLOCK_SIZE + threadIdx.x < N && Row < N)
      As[threadIdx.y][threadIdx.x] = a[Row*N + k*BLOCK_SIZE + threadIdx.x];
    else
      As[threadIdx.y][threadIdx.x] = 0.0;

    if (k*BLOCK_SIZE + threadIdx.y < N && Col < N)
      Bs[threadIdx.y][threadIdx.x] = b[(k*BLOCK_SIZE + threadIdx.y)*N + Col];
    else
      Bs[threadIdx.y][threadIdx.x] = 0.0;

    __syncthreads();

    for (int n = 0; n < BLOCK_SIZE; ++n)
      CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

    __syncthreads();
  }

  if (Row < N && Col < N)
    c[((blockIdx.y * blockDim.y + threadIdx.y)*N) +
      (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
}

void dotProductBlockCUDA (float*a, float*b, float*c, int N) {
  float *Da, *Db, *Dc;

  size_t size = N * N * sizeof(float);

  cudaMalloc(&Da, size);
  cudaMalloc(&Db, size);

  cudaMemcpy(Da, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(Db, b, size, cudaMemcpyHostToDevice);

  cudaMalloc(&Dc, size);

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((N + dimBlock.x -1) / dimBlock.x, (N + dimBlock.y -1 )/ dimBlock.y);

  cudaBlockKernel<<<dimGrid, dimBlock>>>(Da, Db, Dc, N);

  cudaDeviceSynchronize();

  cudaMemcpy(c, Dc, size, cudaMemcpyDeviceToHost);

  cudaFree(Da);
  cudaFree(Db);
  cudaFree(Dc);
}

void run_dotproduct(float *A, float *B, float *C, int SIZE, void (*f) (float*,float*,float*, int)){
  long long unsigned tt;

  fillMatrices(A,B,C, SIZE);
  //clearCache();

  start();
  f(A,  B,  C, SIZE);
  tt = stop();

  validate_rows(C, SIZE);

  printResults(tt);

}

void papi ( float *A, float *B, float *C, int SIZE, void (*f)(float*, float*, float*, int), char* tag ) {

  fillMatrices(A,B,C, SIZE);
  //clearCache();

  int i;
  PAPI_start_counters( PAPI_events, NUM_EVENTS );

  f(A,B,C,SIZE);

  PAPI_stop_counters( counters, NUM_EVENTS );

  if (!strcmp("mr",tag) ) {
    printf("%f", (counters[1]*100.0f)/(counters[0]*1.0f) );
  }
  else {
    for (i=0; i<NUM_EVENTS; i++)
      printf("%lld",counters[i]);
  }
  printf("\n");
}

int main (int argc, char *argv[]) {

  float *A, *B, *C;
  int SIZE = atoi(argv[2]);

  A = (float *) malloc( SIZE * SIZE * sizeof(float)); 
  B = (float *) malloc( SIZE * SIZE * sizeof(float)); 
  C = (float *) malloc( SIZE * SIZE * sizeof(float)); 

  void (*implement) (float *,float *,float *, int);

  if( !strcmp("dotProductCUDA",argv[1]) ){
    implement = dotProductCUDA;
  } else if( !strcmp("dotProductBlockCUDA",argv[1]) ){
    implement = dotProductBlockCUDA;
  } else if( !strcmp("comCUDA",argv[1]) ){
    implement = comCUDA;
  } else {
    return 1;
  }

  if ( !strcmp("time",argv[3]) ) {
    run_dotproduct(A,B,C,SIZE,implement);
  } else {
    PAPI_library_init(PAPI_VER_CURRENT);

    if ( !strcmp("mrl1",argv[3]) ) {
      NUM_EVENTS = 2;
      counters = (long long *) malloc (NUM_EVENTS *sizeof(long long) );
      PAPI_events = (int*) malloc (sizeof(int) * NUM_EVENTS);
      PAPI_events[0] = PAPI_LD_INS;
      PAPI_events[1] = PAPI_L2_DCR;
    } else if ( !strcmp("mrl2",argv[3]) ) {
      NUM_EVENTS = 2;
      counters = (long long *) malloc (NUM_EVENTS *sizeof(long long) );
      PAPI_events = (int*) malloc (sizeof(int) * NUM_EVENTS);
      PAPI_events[0] = PAPI_L2_DCR;
      PAPI_events[1] = PAPI_L3_DCR;	
    } else if ( !strcmp("mrl3",argv[3]) ) {
      NUM_EVENTS = 2;
      counters = (long long *) malloc (NUM_EVENTS *sizeof(long long) );
      PAPI_events = (int*) malloc (sizeof(int) * NUM_EVENTS);
      PAPI_events[0] = PAPI_L3_TCA;
      PAPI_events[1] = PAPI_L3_TCM;
    } else if ( !strcmp("L3_TCM",argv[3]) ) {
      NUM_EVENTS = 1;
      counters = (long long *) malloc (NUM_EVENTS *sizeof(long long) );
      PAPI_events = (int*) malloc (sizeof(int) * NUM_EVENTS);
      PAPI_events[0] = PAPI_L3_TCM;
    } else if ( !strcmp("FP_INS",argv[3]) ) {
      NUM_EVENTS = 1;
      counters = (long long *) malloc (NUM_EVENTS *sizeof(long long) );
      PAPI_events = (int*) malloc (sizeof(int) * NUM_EVENTS);
      PAPI_events[0] = PAPI_FP_INS;
    } else {
      return 1;
    }

    papi( A,B,C,SIZE, implement, argv[3] );
  }

  return 0;
}
