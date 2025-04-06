#include <stdio.h>
#include <cmath>

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
    int N = 10;
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
        float intermediate[curSize];
        cudaMemcpy(&intermediate , d_A, curSize * sizeof(float), cudaMemcpyDeviceToHost);
        
        curSize = ceil(curSize / 2.0);
        halfSize = ceil(curSize / 2.0);

        printf("Intermediate sum (size=%d):\n", curSize);
        for (int i = 0; i < curSize; i++) {
            printf("%.2f ", intermediate[i]);
        }
        printf("\n");
    }

    cudaDeviceSynchronize();

    float result;
    cudaMemcpy(&result, d_A, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Input array: ");
    for (int i = 0; i < N; i++) {
        printf("%.2f ", h_A[i]);
    }
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

