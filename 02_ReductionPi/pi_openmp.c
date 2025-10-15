// pi_openmp.c
// gcc -O3 -fopenmp pi_openmp.c -o pi_openmp
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double pi_serial(long n) {
    double step = 1.0 / (double)n;
    double sum = 0.0;
    for (long i = 0; i < n; ++i) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    return step * sum;
}

double pi_parallel(long n) {
    double step = 1.0 / (double)n;
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (long i = 0; i < n; ++i) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    return step * sum;
}

int main(int argc, char** argv) {
    long n = (argc > 1) ? atol(argv[1]) : 200000000L; // 2e8 by default
    printf("Iterations: %ld\n", n);

    double t0 = omp_get_wtime();
    double pi1 = pi_serial(n);
    double t1 = omp_get_wtime();

    double t2 = omp_get_wtime();
    double pi2 = pi_parallel(n);
    double t3 = omp_get_wtime();

    double ts = t1 - t0, tp = t3 - t2;
    printf("Serial   π = %.12f  time = %.3f s\n", pi1, ts);
    printf("Parallel π = %.12f  time = %.3f s\n", pi2, tp);
    printf("Speedup = %.2fx\n", ts / tp);
    return 0;
}
