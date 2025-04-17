#include <stdio.h>

__global__ void elementwiseSum(
    float* A, float* B, float* C
) {
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main() {
    const int N = 10;
    float h_A[N], h_B[N], h_C[N];
    
    // Initialize with random values
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    // create array on device
    // Cannot create array with d_A[N] here because it'll create a fixed
    // array on host memory, which cannot be redirected to device.
    float *d_A, *d_B, *d_C;
    // cudaMalloc needs a pointer to pointer because it needs to 
    // modify the pointer value
    cudaMalloc((void**) &d_A, N * sizeof(float));
    cudaMalloc((void**) &d_B, N * sizeof(float));
    cudaMalloc((void**) &d_C, N * sizeof(float));

    // cudaMemcpy takes a pointer, but since we allocate an array, it's already
    // a pointer
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    elementwiseSum<<<1, N>>>(d_A, d_B, d_C);

    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    printf("Elementwise Sum Output:\n");
    printf("A: ");
    for (int i = 0; i < N; i++) {
        printf("%.2f ", h_A[i]);
    }
    printf("\nB: ");
    for (int i = 0; i < N; i++) {
        printf("%.2f ", h_B[i]);
    }
    printf("\nC: ");
    for (int i = 0; i < N; i++) {
        printf("%.2f ", h_C[i]);
    }
    printf("\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

} 