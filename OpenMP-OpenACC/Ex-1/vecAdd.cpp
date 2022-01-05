#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#if _OPENMP
#include <omp.h>
#elif _OPENACC
#include <openacc.h>
#endif
 
#define VECSIZE 100000

using namespace std::chrono;


void vec_add_gpu(double *a, double *b, double *c, int n){
#if _OPENMP
#pragma omp target teams distribute parallel for \
    map(to: a[:n], b[:n]) map(from: c[:n])
#elif _OPENACC
#pragma acc parallel loop gang vector \
    copyin(a[:n]) copyout(c[:n])
#endif
    for(int i = 0; i < n; ++i)
        c[i] = a[i] + b[i];
}

int main( int argc, char* argv[] )
{
#if _OPENMP
    printf("Running the OpenMP version \n");
#elif _OPENACC
    printf("Running the OpenACC version \n");
#else
    printf("Running the Sequential version \n");
#endif

    // Size of vectors
    int n = VECSIZE;

    printf("Adding vectors of size %d \n", n);
 
    // Host input vectors
    double *a;
    double *b;
    //Host output vector
    double *c;
 

    // Allocate memory for each vector on host
    a = new double [VECSIZE];//(double*)malloc(bytes);
    b = new double [VECSIZE];
    c = new double [VECSIZE];

    // Initialize vectors on host
    for( int i = 0; i < n; ++i ) {
        a[i] = sin(i)*sin(i);
        b[i] = cos(i)*cos(i);
    }

    time_point<system_clock> start;
    start = system_clock::now();
 
    vec_add_gpu(a, b, c, n);

    duration<double> elapsed = system_clock::now() - start;

    // Sum up vector c and print result divided by n, this should equal 1 within error
    double sum = 0;
    for(int i=0; i<n; ++i)
        sum += c[i];
    printf("final result: %f\n", sum/(double)n);

    printf("Time taken T[secs] = %f \n", elapsed.count());

    // Release host memory
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
