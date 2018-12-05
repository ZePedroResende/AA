#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <papi.h>
#include <omp.h>
#include <sys/time.h>

#define BLOCK_SIZE 32

int NUM_EVENTS=0;
int *PAPI_events;
long long *counters;

void pm (float*mat, int N) {
	int i,j;
	for(i=0;i<N;i++) {
		for(j=0;j<N;j++)
			printf("  %10.2f",mat[i*N+j]);
		printf("\n");
	}
}

void papiCount ( float *a, float *b, float *c, int N, void (*func)(float*, float*, float*, int), char* tag ) {
	int i;
	PAPI_start_counters( PAPI_events, NUM_EVENTS );
	
	func (a,b,c,N);
	
	PAPI_stop_counters( counters, NUM_EVENTS );
	
	if (!strncmp("mr",tag,2) ) {
		printf("%f", (counters[1]*100.0f)/(counters[0]*1.0f) );
	}
	else {
		for (i=0; i<NUM_EVENTS; i++)
			printf("%lld",counters[i]);
	}
	printf("\n");
}

void timer ( float *a, float *b, float *c, int N, void (*func)(float*, float*, float*, int) ) {
	struct timeval start, end;
	gettimeofday(&start, 0);

	func (a,b,c,N);
	
	gettimeofday(&end, 0);
	long long elapsed = (end.tv_sec-start.tv_sec)*1000000LL + end.tv_usec-start.tv_usec;
	
	printf("%lld\n", elapsed);
}

void transpose (float*matrix, int N) {
	int i,j;
	float tmp;
	for (i=0; i<N; ++i)
		for (j=0; j<i; ++j) {
			tmp = matrix[i*N+j];
			matrix[i*N+j] = matrix[j*N+i];
			matrix[j*N+i] = tmp;
		}
}

void transposeOMP (float*matrix, int N) {
	int i,j;
	float tmp;
	#pragma omp parallel for schedule(dynamic) num_threads(20)
	for (i=0; i<N; ++i) {
		for (j=0; j<i; ++j) {
			tmp = matrix[i*N+j];
			matrix[i*N+j] = matrix[j*N+i];
			matrix[j*N+i] = tmp;
		}
	}
}

float* randomMatrix(int N) {
	int i,j;
	float*matrix = (float*) malloc ( N *N *sizeof(float) );
	srand(time(NULL));
	for (i=0; i<N; ++i)
		for (j=0;j<N;j++)
			matrix[i*N+j] = ((float)rand()/(float)(RAND_MAX)) * 100 ;
	return matrix;
}

float* unitaryMatrix(int N) {
	int i;
	float*matrix = (float*) malloc ( N *N *sizeof(float) );
	for (i=0; i<N*N; ++i)
		matrix[i] = 1;
	return matrix;
}

float* emptyMatrix(int N) {
	float* matrix = (float*) calloc (N*N, sizeof(float));
	return matrix;
}

void dotProductA (float*a, float*b, float*c, int N) {
	int i,j,k;
	for (i=0; i<N; ++i)
		for (j=0; j<N; ++j)
			for (k=0; k<N; ++k)
				c[i*N+j] += a[i*N+k] * b[k*N+j]; 
}

void dotProductB (float* __restrict__ a, float* __restrict__ b, float* __restrict__ c, int N) {
	int i,k,j;
	for (i=0; i<N; ++i)
		for (k=0; k<N; ++k)
			for (j=0; j<N; ++j)
				c[i*N+j] += a[i*N+k] * b[k*N+j]; 
}

void dotProductC (float*a, float*b, float*c, int N) {
	int j,k,i;
	for (j=0; j<N; ++j)
		for (k=0; k<N; ++k)
			for (i=0; i<N; ++i)
				c[i*N+j] += a[i*N+k] * b[k*N+j]; 
}

void dotProductAtranspose (float*a, float*b, float*c, int N) {
	int i,j,k;
	transpose(b,N);
	for (i=0; i<N; ++i)
		for (j=0; j<N; ++j)
			for (k=0; k<N; ++k)
				c[i*N+j] += a[i*N+k] * b[j*N+k]; 
}

void dotProductCtranspose (float* __restrict__ a, float* __restrict__ b, float* __restrict__ c, int N) {
	int j,k,i;
	transpose(a,N);
	transpose(b,N);
	for (j=0; j<N; ++j)
		for (k=0; k<N; ++k)
			for (i=0; i<N; ++i)
				c[j*N+i] += a[k*N+i] * b[k*N+j];
	transpose(c,N);
}

void dotProductBlock (float*a, float*b, float*c, int N) {
	int blSize = 16;
	int blRow,blCol,i,k,j;
	transpose(b,N);
	for(blRow=0; blRow<N; blRow+= blSize) {
		for(blCol=0; blCol<N; blCol+= blSize) {
			for(i=0;i<N;i++) {
				for(j = blRow; j<((blRow+blSize)>N?N:(blRow+blSize)); j++) {
					float acc = 0;
					for(k = blCol; k<((blCol+blSize)>N?N:(blCol+blSize)); k++) {
						acc += a[i*N+k]*b[k*N+j];
					}
					c[i*N+j] += acc;
				}
			}
		}
	}
}

void dotProductBlockOMP (float*a, float*b, float*c, int N) {
	int blSize = 16;
	int blRow,blCol,i,k,j;
	transposeOMP(b,N);

	#pragma omp parallel for schedule(static) collapse(2) num_threads(20)
	for(blRow=0; blRow<N; blRow+= blSize) {
		for(blCol=0; blCol<N; blCol+= blSize) {
			for(i=0;i<N;i++) {
				for(j = blRow; j<((blRow+blSize)>N?N:(blRow+blSize)); j++) {
					float acc = 0;
					for(k = blCol; k<((blCol+blSize)>N?N:(blCol+blSize)); k++) {
						acc += a[i*N+k]*b[k*N+j];
					}
					c[i*N+j] += acc;
				}
			}
		}
	}
}

__global__ void cudaKernel (float *Da, float *Db, float *Dc, int N) {
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;
	int k;
	if (i>=N || j>=N)return;
	float res=0;
	for(k=0; k<N; ++k)
		res += Da[i*N + k] * Db[k*N + j];
	Dc[i*N + j]=res;
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

int main(int argc, char *argv[]) {
	if ( argc < 4 ) {
		printf("Not enought arguments\n");
		return 1;
	}
	
	float *a, *b, *c;
	int N = atoi( argv[1] );
	a = randomMatrix(N);
	b = unitaryMatrix(N);
	c = emptyMatrix(N);
	void (*func) (float*, float*, float*, int);
	
	if( !strcmp("ijk",argv[2]) )
		func = dotProductA;

	else if( !strcmp("ikj",argv[2]) )
		func = dotProductB;

	else if( !strcmp("jki",argv[2]) )
		func = dotProductC;

	else if( !strcmp("ijk_t",argv[2]) )
		func = dotProductAtranspose;

	else if( !strcmp("jki_t",argv[2]) )
		func = dotProductCtranspose;

	else if( !strcmp("block",argv[2]) )
		func = dotProductBlock;

	else if( !strcmp("block_omp",argv[2]) )
		func = dotProductBlockOMP;

	else if( !strcmp("cuda",argv[2]) ) {
		func = dotProductCUDA;
		cudaSetDevice(0);
	}
	else if( !strcmp("block_cuda",argv[2]) ) {
		func = dotProductBlockCUDA;
		cudaSetDevice(0);
	}
	else {
		printf("Wrong dot product funcion. Avaiable:\n\tijk\n\tikj\n\tjki\nijk_t\n\tjki_t\n\tblock\n\tblock_omp\n\tcuda\n\tblock_cuda\n");
		return 1;
	}
	
	char **clearcache = (char**) malloc(100 * sizeof(char*) );
	int i,j;
	for (i=0; i<100; i++) {  
		clearcache[i] = (char*)malloc(26*1024*1024);
		for (j = 0; j < 26*1024*1024; ++j)
			clearcache[i][j] = i+j;	
	}

	if (argc>5) {
		pm(a,N); printf("\n"); pm(b,N); printf("\n");
	}

	if ( !strcmp("time",argv[3]) ) {
		timer ( a,b,c,N, func );
	}
	else {
		PAPI_library_init(PAPI_VER_CURRENT);
		if ( !strcmp("mrl1",argv[3]) ) {
			NUM_EVENTS = 2;
			counters = (long long *) malloc (NUM_EVENTS *sizeof(long long) );
			PAPI_events = (int*) malloc (sizeof(int) * NUM_EVENTS);
			PAPI_events[0] = PAPI_LD_INS;
			PAPI_events[1] = PAPI_L2_DCR;
		}
		else if ( !strcmp("mrl2",argv[3]) ) {
			NUM_EVENTS = 2;
			counters = (long long *) malloc (NUM_EVENTS *sizeof(long long) );
			PAPI_events = (int*) malloc (sizeof(int) * NUM_EVENTS);
			PAPI_events[0] = PAPI_L2_DCR;
			PAPI_events[1] = PAPI_L3_DCR;	
		}
		else if ( !strcmp("mrl3",argv[3]) ) {
			NUM_EVENTS = 2;
			counters = (long long *) malloc (NUM_EVENTS *sizeof(long long) );
			PAPI_events = (int*) malloc (sizeof(int) * NUM_EVENTS);
			PAPI_events[0] = PAPI_L3_TCA;
			PAPI_events[1] = PAPI_L3_TCM;
		}
		else if ( !strcmp("L3_TCM",argv[3]) ) {
			NUM_EVENTS = 1;
			counters = (long long *) malloc (NUM_EVENTS *sizeof(long long) );
			PAPI_events = (int*) malloc (sizeof(int) * NUM_EVENTS);
			PAPI_events[0] = PAPI_L3_TCM;
		}
		else if ( !strcmp("FP_INS",argv[3]) ) {
			NUM_EVENTS = 1;
			counters = (long long *) malloc (NUM_EVENTS *sizeof(long long) );
			PAPI_events = (int*) malloc (sizeof(int) * NUM_EVENTS);
			PAPI_events[0] = PAPI_FP_INS;
		}
		else {
			printf("Wrong counter. Avaiable:\n\ttime\n\tmrl1\n\tmrl2\n\tmrl3\n\tL3_TCM\n\tFP_INS\n");
			return 1;
		}
		papiCount ( a,b,c,N, func, argv[3] );
	}

	if (argc > 4)
		pm(c,N);

	return 0;
}