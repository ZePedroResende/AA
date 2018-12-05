#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cublas_v2.h>


#define N 1024
#define TIME_RESOLUTION 1000000 // time measuring resolution (us)
#define BLOCK_SIZE 8

long long unsigned initial_time;
struct timeval t;

void checkCUDAError (const char *msg) {
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		printf("Cuda error: %s, %s\n ", msg, cudaGetErrorString( err));
		exit(-1);
	}
}

long long unsigned mark(){
	gettimeofday(&t, NULL);
    return t.tv_sec * TIME_RESOLUTION + t.tv_usec;
}

void start(void) {
    gettimeofday(&t, NULL);
    initial_time = t.tv_sec * TIME_RESOLUTION + t.tv_usec;
}

long long unsigned stop(void) {
    gettimeofday(&t, NULL);
    long long unsigned final_time = t.tv_sec * TIME_RESOLUTION + t.tv_usec;

    return final_time - initial_time;
}

int validate_rows(float c[N*N]){
    int i, j, r = 1;
    for (i = 0; i < N && r; i++){
        for (j = 0; j < N && r; j++){
            if (c[(i*N)+j] != c[i*N]) r = 0;
        }
    }
    return r;
}

int validate_columns(float c[N][N]){
    int i, j, r = 1;
    for (i = 0; i < N && r; i++){
        for (j = 0; j < N && r; j++){
            if (c[i][j] != c[0][j]) r = 0;
        }
    }
    return r;
}



__global__ void cuda_mm_kernel(float *ad, float *bd, float *cd, int n){
	int j;
    int i = threadIdx.x;
    int k = blockIdx.x;

	__shared__ int local_array[N];
	local_array[i] = 0;
    for (j = 0; j < n; j++){
        //cd[(i*N)+j] += ad[(i*N)+k] * bd[(k*N)+j];
        local_array[i] += ad[(i*N)+k] * bd[(k*N)+j];
    }

    //__syncthreads();

    //if (i == 0){
    //	for (i = 0; i < n; i++) for (j = 0; j < n; j++) cd[(i*N)+j] += local_array[i];
    //}
    for (j = 0; j < n; j++) cd[(i*N)+j] = local_array[i];

}

void cuda_mm(float c[N*N], float a[N*N], float b[N*N], int n) {
    float *ad, *bd, *cd;
    int size = n * n * sizeof(float);
    long long unsigned before, after;
    //int n_threads = n * n;
    before = mark();
    cudaMalloc((void**)&ad, size);
    cudaMalloc((void**)&bd, size);
    cudaMalloc((void**)&cd, size);
    checkCUDAError("mem allocation");

    cudaMemcpy(ad, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(bd, b, size, cudaMemcpyHostToDevice);
    //cudaMemcpy(cd, c, size, cudaMemcpyHostToDevice);

    checkCUDAError("memcpy h->d");
    before = mark() - before;

    after = mark();
    cuda_mm_kernel <<< N, N >>>(ad, bd, cd, n);
    checkCUDAError("kernel invocation");

    cudaMemcpy(c, cd, size, cudaMemcpyDeviceToHost);
    checkCUDAError("memcpy d->h");
 
    cudaFree(ad);
    cudaFree(bd);
    cudaFree(cd);
    checkCUDAError("mem free");
    after = mark() - after;

    printf("Before: %llu usecs\nAfter: %llu usecs\nTotal: %llu\n", before, after, before+after);

}


    float a[N*N];
    float b[N*N];
    float c[N*N];

int main(int argc, char* argv[]){


    int i, j;
    for (i = 0; i < N; i++){
        for (j = 0; j < N; j++){
            a[i*N+j] = ((float) rand()) / ((float) RAND_MAX);
            b[i*N+j] = 1.0f;
            c[i*N+j] = 0.0f;
        }
    }
    cuda_mm(c, a, b, N);

    //Calculated a*b. Validate. 
    if (!validate_rows(c)) printf("Bad result\n");
    printf("Done\n");
   

}