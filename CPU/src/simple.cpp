#include <iostream>
#include <cstdlib>
#include <sys/time.h>

#define SIZE 1024// Only powers of 2 to simplify the code
#define TILE_SIZE 32 // the size of the square tile
#define TIME_RESOLUTION 1000000	// time measuring resolution (us)

using namespace std;

float m1[SIZE][SIZE], m2[SIZE][SIZE], result[SIZE][SIZE];

double clearcache [30000000];
long long unsigned initial_time;
timeval t;


void clearCache (void) {
  for (unsigned i = 0; i < 30000000; ++i)
    clearcache[i] = i;
}

void printResults (long long unsigned tt) {
  cout << "Execution time: " << tt << " usecs" << endl;
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

void fillMatrices (float a[][SIZE],float b[][SIZE],float r[][SIZE]) {
  for (unsigned i = 0; i < SIZE; ++i) {
    for (unsigned j = 0; j < SIZE; ++j) {
      a[i][j] = ((float) rand()) / ((float) RAND_MAX);
      a[i][j] = 1;
      r[i][j] = 0;
    }
  }
}

void ijk(float a[][SIZE],float b[][SIZE],float c[][SIZE], unsigned size) {
  for (unsigned i = 0; i < size; ++i) {
    for (unsigned j = 0; j < size; ++j) {
      for (unsigned k = 0; k < size; ++k) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
}

void ikj(float a[][SIZE],float b[][SIZE],float c[][SIZE], unsigned size) {
  for (unsigned i = 0; i < size; ++i) {
    for (unsigned k = 0; k < size; ++k) {
      for (unsigned j = 0; j < size; ++j) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
}

void jki(float a[][SIZE],float b[][SIZE],float c[][SIZE], unsigned size) {
  for (unsigned j = 0; j < size; ++j) {
    for (unsigned k = 0; k < size; ++k) {
      for (unsigned i = 0; i < size; ++i) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
}

void ijk_transposed(float a[][SIZE],float b[][SIZE],float c[][SIZE], unsigned size) {
  for (unsigned i = 0; i < size; ++i) {
    for (unsigned j = 0; j < size; ++j) {
      for (unsigned k = 0; k < size; ++k) {
        c[j][i] += a[i][k] * b[j][k];
      }
    }
  }
}

void ikj_transposed(float a[][SIZE],float b[][SIZE],float c[][SIZE], unsigned size) {
  for (unsigned i = 0; i < size; ++i) {
    for (unsigned k = 0; k < size; ++k) {
      for (unsigned j = 0; j < size; ++j) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
}

void jki_transposed(float a[][SIZE],float b[][SIZE],float c[][SIZE], unsigned size) {
  for (unsigned j = 0; j < size; ++j) {
    for (unsigned k = 0; k < size; ++k) {
      for (unsigned i = 0; i < size; ++i) {
        c[j][i] += a[k][i] * b[j][k];
      }
    }
  }
}

void run_dotproduct(void (*f) (float [][SIZE],float [][SIZE],float [][SIZE], unsigned)){

  long long unsigned tt;

  fillMatrices(m1,m2,result);
  clearCache();
  start();
  f(m1,  m2,  result, SIZE);
  tt = stop();
  printResults(tt);

}

int main (int argc, char *argv[]) {

  run_dotproduct(ijk);
  run_dotproduct(ikj);
  run_dotproduct(jki);
  run_dotproduct(ijk_transposed);
  run_dotproduct(ikj_transposed);
  run_dotproduct(jki_transposed);

  return 0;
}
