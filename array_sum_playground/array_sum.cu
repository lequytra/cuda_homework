#include <stdio.h>
#include <cmath>

// FINDINGS:
// - Surprisingly, padding N to power of 2 to make sure all threads are utilized 
//      overall decrease performance.
// - The current parallization scheme only parallelize across threads within
//      a single block, so if array gets too large (> 1024 * 2), we start
//      losing data.

// nvcc -o array_sum array_sum.cu && ./array_sum

__global__ void computeSum(
    float* array, int halfSize, int size
) {
    int i = threadIdx.x; 
    if (i + halfSize < size) {
        array[i] = array[i] + array[i + halfSize]; 
    }
}

int main() {
    srand(time(NULL)); // Set random seed based on current time
    int N = rand() % (1024 * 2) + 10; // Random N between 10 and 100
    float h_A[N];
    // Initialize array with random values
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
    }

    float* d_A;
    cudaMalloc((void**) &d_A, N * sizeof(float));
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);

    int halfSize = ceil(N / 2.0); 
    int curSize = N; 

    while (curSize > 1) {
        computeSum<<<1, halfSize>>>(d_A, halfSize, curSize);
        curSize = halfSize;
        halfSize = ceil(curSize / 2.0);
    }

    cudaDeviceSynchronize();

    float result;
    cudaMemcpy(&result, d_A, sizeof(float), cudaMemcpyDeviceToHost);

    // printf("Input array: ");
    // for (int i = 0; i < N; i++) {
    //     printf("%.2f ", h_A[i]);
    // }
    printf("\nN: %.2d\n", N);
    printf("\nSum: %.2f\n", result);

    // Calculate expected sum on CPU for verification
    float expected_sum = 0.0f;
    for (int i = 0; i < N; i++) {
        expected_sum += h_A[i];
    }
    printf("Expected sum: %.2f\n", expected_sum);
    printf("Difference: %.2f\n", fabs(expected_sum - result));

    cudaFree(d_A);

}

