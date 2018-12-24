#include <iostream>
#include <string.h>
#include <cstdlib>
#include <sys/time.h>
#include "papi.h"
#define TILE_SIZE 128 // the size of the square tile
#define TIME_RESOLUTION 1000000	// time measuring resolution (us)


using namespace std;


double clearcache [30000000];
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
  cout <<tt<< endl;
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

void fillMatrices (float **A,float **B,float **C, int SIZE) {

  for (int i = 0; i < SIZE; ++i) {
    for (int j = 0; j < SIZE; ++j) {
      A[i][j] = ((float) rand()) / ((float) RAND_MAX);
      B[i][j] = 1;
      C[i][j] = 0;
    }
  }
}

void transpose (float **matrix, int N) {
    int i,j;
    float tmp;
    for (i=0; i<N; ++i)
        for (j=0; j<i; ++j) {
            tmp = matrix[i][j];
            matrix[i][j] = matrix[j][i];
            matrix[j][i] = tmp;
        }
}

void ijk(float **A, float **B, float **C, int SIZE) {
  for (int i = 0; i < SIZE; ++i) {
    for (int j = 0; j < SIZE; ++j) {
      for (int k = 0; k < SIZE; ++k) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

void ikj(float **A, float **B, float **C, int SIZE) {
  for (int i = 0; i < SIZE; ++i) {
    for (int k = 0; k < SIZE; ++k) {
      for (int j = 0; j < SIZE; ++j) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

void jki(float **A, float **B, float **C, int SIZE) {
  for (int j = 0; j < SIZE; ++j) {
    for (int k = 0; k < SIZE; ++k) {
      for (int i = 0; i < SIZE; ++i) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

void ijk_transposed(float **A, float **B, float **C, int SIZE) {
  transpose(B,SIZE);
  for (int i = 0; i < SIZE; ++i) {
    for (int j = 0; j < SIZE; ++j) {
      for (int k = 0; k < SIZE; ++k) {
        C[i][j] += A[i][k] * B[j][k];
      }
    }
  }
}

void jki_transposed(float **A, float **B, float **C, int SIZE) {
  transpose(A,SIZE);
  transpose(B,SIZE);
  for (int j = 0; j < SIZE; ++j) {
    for (int k = 0; k < SIZE; ++k) {
      for (int i = 0; i < SIZE; ++i) {
        C[j][i] += A[k][i] * B[j][k];
      }
    }
  }
  transpose(C,SIZE);
}

void block(float** __restrict__ A, float** __restrict__ B, float** __restrict__ C, int SIZE) {
	int i = 0, j = 0, k = 0, jj = 0, kk = 0;
  int block_size = 16;
  float tmp;

  for (jj = 0; jj < SIZE; jj += block_size) {
    for (kk = 0; kk < SIZE; kk += block_size) {
      for (i = 0; i < SIZE; i++) {
        for (j = jj; j < ((jj + block_size) > SIZE ? SIZE : (jj + block_size)); j++) {
          tmp = 0.0f;
          for (k = kk; k < ((kk + block_size) > SIZE ? SIZE : (kk + block_size)); k++) {
            tmp += A[i][k] * B[k][j];
          }
          C[i][j] += tmp;
        }
      }
    }
  }
}

void blockOMP1(float **A, float **B, float **C, int SIZE) {
	int i = 0, j = 0, k = 0, jj = 0, kk = 0;
  int block_size = 16;
  float tmp;

#pragma omp parallel for schedule(static,2) shared(A, B, C, SIZE, block_size) private(i, j, k, jj, kk, tmp)
  for (jj = 0; jj < SIZE; jj += block_size) {
    for (kk = 0; kk < SIZE; kk += block_size) {
      for (i = 0; i < SIZE; i++) {
        for (j = jj; j < ((jj + block_size) > SIZE ? SIZE : (jj + block_size)); j++) {
          tmp = 0.0f;
          for (k = kk; k < ((kk + block_size) > SIZE ? SIZE : (kk + block_size)); k++) {
            tmp += A[i][k] * B[k][j];
          }
          C[i][j] += tmp;
        }
      }
    }
  }
}

void blockOMP2(float **A, float **B, float **C, int SIZE) {
	int i = 0, j = 0, k = 0, jj = 0, kk = 0;
  int block_size = 16;
  float tmp;

#pragma omp parallel shared(A, B, C, SIZE, block_size) private(i, j, k, jj, kk, tmp)
{

#pragma omp for schedule(static,2)
  for (jj = 0; jj < SIZE; jj += block_size) {
    for (kk = 0; kk < SIZE; kk += block_size) {
      for (i = 0; i < SIZE; i++) {
        for (j = jj; j < ((jj + block_size) > SIZE ? SIZE : (jj + block_size)); j++) {
          tmp = 0.0f;
          for (k = kk; k < ((kk + block_size) > SIZE ? SIZE : (kk + block_size)); k++) {
            tmp += A[i][k] * B[k][j];
          }
          C[i][j] += tmp;
        }
      }
    }
  }
}
}

void blockOMP3(float **A, float **B, float **C, int SIZE) {
	int i = 0, j = 0, k = 0, jj = 0, kk = 0;
  int block_size = 16;
  float tmp;

#pragma omp parallel for schedule(static) collapse(2) num_threads(24)
  for (jj = 0; jj < SIZE; jj += block_size) {
    for (kk = 0; kk < SIZE; kk += block_size) {
      for (i = 0; i < SIZE; i++) {
        for (j = jj; j < ((jj + block_size) > SIZE ? SIZE : (jj + block_size)); j++) {
          tmp = 0.0f;
          for (k = kk; k < ((kk + block_size) > SIZE ? SIZE : (kk + block_size)); k++) {
            tmp += A[i][k] * B[k][j];
          }
          C[i][j] += tmp;
        }
      }
    }
  }
}

void blockOMP4(float **A, float **B, float **C, int SIZE) {
	int i = 0, j = 0, k = 0, jj = 0, kk = 0;
  int block_size = 16;
  float tmp;

#pragma omp parallel for private(i, j, k, jj, kk, tmp)
  for (jj = 0; jj < SIZE; jj += block_size) {
    for (kk = 0; kk < SIZE; kk += block_size) {
      for (i = 0; i < SIZE; i++) {
        for (j = jj; j < ((jj + block_size) > SIZE ? SIZE : (jj + block_size)); j++) {
          tmp = 0.0f;
          for (k = kk; k < ((kk + block_size) > SIZE ? SIZE : (kk + block_size)); k++) {
            tmp += A[i][k] * B[k][j];
          }
          C[i][j] += tmp;
        }
      }
    }
  }
}

void blockVec(float **A, float **B, float **C, int SIZE) {
	int i = 0, j = 0, jj = 0, kk = 0, z = 0;
  int block_size = 16;
  float temp[16], aux[16];

  for (jj = 0; jj < SIZE; jj += block_size) {
    for (kk = 0; kk < SIZE; kk += block_size) {
      for (i = 0; i < SIZE; i++) {
        for (j = jj, z = 0; j < ((jj + block_size) > SIZE ? SIZE : (jj + block_size)); j++, z++) {
          temp[0] = A[i][kk] * B[kk][j];
          temp[1] = A[i][kk+1] * B[kk+1][j];
          temp[2] = A[i][kk+2] * B[kk+2][j];
          temp[3] = A[i][kk+3] * B[kk+3][j];
          temp[4] = A[i][kk+4] * B[kk+4][j];
          temp[5] = A[i][kk+5] * B[kk+5][j];
          temp[6] = A[i][kk+6] * B[kk+6][j];
          temp[7] = A[i][kk+7] * B[kk+7][j];
          temp[8] = A[i][kk+8] * B[kk+8][j];
          temp[9] = A[i][kk+9] * B[kk+9][j];
          temp[10] = A[i][kk+10] * B[kk+10][j];
          temp[11] = A[i][kk+11] * B[kk+11][j];
          temp[12] = A[i][kk+12] * B[kk+12][j];
          temp[13] = A[i][kk+13] * B[kk+13][j];
          temp[14] = A[i][kk+14] * B[kk+14][j];
          temp[15] = A[i][kk+15] * B[kk+15][j];

          aux[z] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7] + temp[8] + temp[9] + temp[10] + temp[11] + temp[12] + temp[13] + temp[14] + temp[15];
        }

        for (j = jj, z= 0; j < ((jj + block_size) > SIZE ? SIZE : (jj + block_size)); j++, z++) {
          C[i][j] += aux[z];
        }
      }
    }
  }
}
void blockOMPVec(float **A, float **B, float **C, int SIZE) {
  int i = 0, j = 0, jj = 0, kk = 0;
  int block_size = 16;
  float temp[16];

#pragma omp parallel for private(i, j, jj, kk, temp)
  for (jj = 0; jj < SIZE; jj += block_size) {
    for (kk = 0; kk < SIZE; kk += block_size) {
      for (i = 0; i < SIZE; i++) {

        #pragma omp simd
        for (j = jj; j < ((jj + block_size) > SIZE ? SIZE : (jj + block_size)); j++) {
          temp[0] = A[i][kk] * B[kk][j];
          temp[1] = A[i][kk+1] * B[kk+1][j];
          temp[2] = A[i][kk+2] * B[kk+2][j];
          temp[3] = A[i][kk+3] * B[kk+3][j];
          temp[4] = A[i][kk+4] * B[kk+4][j];
          temp[5] = A[i][kk+5] * B[kk+5][j];
          temp[6] = A[i][kk+6] * B[kk+6][j];
          temp[7] = A[i][kk+7] * B[kk+7][j];
          temp[8] = A[i][kk+8] * B[kk+8][j];
          temp[9] = A[i][kk+9] * B[kk+9][j];
          temp[10] = A[i][kk+10] * B[kk+10][j];
          temp[11] = A[i][kk+11] * B[kk+11][j];
          temp[12] = A[i][kk+12] * B[kk+12][j];
          temp[13] = A[i][kk+13] * B[kk+13][j];
          temp[14] = A[i][kk+14] * B[kk+14][j];
          temp[15] = A[i][kk+15] * B[kk+15][j];

          
          C[i][j] += temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7] + temp[8] + temp[9] + temp[10] + temp[11] + temp[12] + temp[13] + temp[14] + temp[15];
        }
      }
    }
  }
}

void knl(float **A, float **B, float **C, int SIZE) {
	int i = 0, j = 0, jj = 0, kk = 0, z = 0;
  int block_size = 16;
  float temp[16], aux[16];

#pragma omp parallel for private(i,j,jj,kk,temp,aux,z)
  for (jj = 0; jj < SIZE; jj += block_size) {
    for (kk = 0; kk < SIZE; kk += block_size) {
      for (i = 0; i < SIZE; i++) {
        #pragma omp simd
        for (j = jj, z = 0; j < ((jj + block_size) > SIZE ? SIZE : (jj + block_size)); j++, z++) {
          temp[0] = A[i][kk] * B[kk][j];
          temp[1] = A[i][kk+1] * B[kk+1][j];
          temp[2] = A[i][kk+2] * B[kk+2][j];
          temp[3] = A[i][kk+3] * B[kk+3][j];
          temp[4] = A[i][kk+4] * B[kk+4][j];
          temp[5] = A[i][kk+5] * B[kk+5][j];
          temp[6] = A[i][kk+6] * B[kk+6][j];
          temp[7] = A[i][kk+7] * B[kk+7][j];
          temp[8] = A[i][kk+8] * B[kk+8][j];
          temp[9] = A[i][kk+9] * B[kk+9][j];
          temp[10] = A[i][kk+10] * B[kk+10][j];
          temp[11] = A[i][kk+11] * B[kk+11][j];
          temp[12] = A[i][kk+12] * B[kk+12][j];
          temp[13] = A[i][kk+13] * B[kk+13][j];
          temp[14] = A[i][kk+14] * B[kk+14][j];
          temp[15] = A[i][kk+15] * B[kk+15][j];

          aux[z] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7] + temp[8] + temp[9] + temp[10] + temp[11] + temp[12] + temp[13] + temp[14] + temp[15];
        }

        for (j = jj, z= 0; j < ((jj + block_size) > SIZE ? SIZE : (jj + block_size)); j++, z++) {
          C[i][j] += aux[z];
        }
      }
    }
  }
}

void validate_rows(float **C, int SIZE){
  int i, j, r = 1;

  for (i = 0; i < SIZE && r; i++){
    for (j = 0; j < SIZE && r; j++){
      if (C[i][j] != C[i][0]) r = 0;
    }
  }

  if(!r) {
    printf("ERRRO");
  }
}

void validate_columns(float **C, int SIZE){
  int i, j, r = 1;

  for (i = 0; i < SIZE && r; i++){
    for (j = 0; j < SIZE && r; j++){
      if (C[i][j] != C[0][j]) r = 0;
    }
  }

  if(!r) {
    printf("ERRRO");
  }
}

void run_dotproduct(float **A, float **B, float **C, int SIZE, void (*f) (float**,float**,float**, int)){

  long long unsigned tt;

  fillMatrices(A,B,C, SIZE);
  clearCache();

  start();
  f(A,  B,  C, SIZE);
  tt = stop();

  validate_rows(C, SIZE);

  //fillMatrices(A,B,C, SIZE);
  //clearCache();

  //f(A,B,C, SIZE);

  //validate_columns(C, SIZE);

  printResults(tt);

}


void papi ( float **A, float **B, float **C, int SIZE, void (*f)(float**, float**, float**, int), char* tag ) {

  fillMatrices(A,B,C, SIZE);
  clearCache();

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

  float **A, **B, **C;
  int SIZE = atoi(argv[2]);

  int k;

  A= (float **) malloc(SIZE * sizeof(float *)); 
  for(k = 0; k < SIZE; k++){
    A[k] = (float *) malloc(SIZE * sizeof(float)); 
  }

  B = (float **) malloc(SIZE * sizeof(float *)); 
  for(k = 0; k < SIZE; k++){
    B[k] = (float *) malloc(SIZE * sizeof(float)); 
  }

  C = (float **) malloc(SIZE * sizeof(float *)); 
  for(k = 0; k < SIZE; k++){
    C[k] = (float *) malloc(SIZE * sizeof(float )); 
  }

  void (*implement) (float **,float **,float **, int);

  if( !strcmp("ijk",argv[1]) )
		implement = ijk;

	else if( !strcmp("ikj",argv[1]) )
		implement = ikj;

	else if( !strcmp("jki",argv[1]) )
		implement = jki;

	else if( !strcmp("ijk_transposed",argv[1]) )
		implement = ijk_transposed;

	else if( !strcmp("jki_transposed",argv[1]) )
		implement = jki_transposed;

	else if( !strcmp("block",argv[1]) )
		implement = block;

	else if( !strcmp("blockOMP1",argv[1]) )
		implement = blockOMP1;

	else if( !strcmp("blockOMP2",argv[1]) )
		implement = blockOMP2;

	else if( !strcmp("blockOMP3",argv[1]) )
		implement = blockOMP3;

	else if( !strcmp("blockOMP4",argv[1]) )
		implement = blockOMP4;

	else if( !strcmp("blockOMPVec",argv[1]) )
		implement = blockOMPVec;

	else if( !strcmp("blockVec",argv[1]) )
		implement = blockVec;

	else if( !strcmp("knl",argv[1]) )
		implement = knl;
	else {
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
			return 1;
		}
		papi( A,B,C,SIZE, implement, argv[3] );
	}

  return 0;
}
