#include <stdio.h>
#include <cmath>

// nvcc -o array_sum_v2 array_sum_v2.cu && ./array_sum_v2

const int MAX_THREADS_PER_BLOCK = 1024;


__global__ void computeSum(
    float* input, float* output, int size
) {
    int defaultHalfSize = blockDim.x; 
    int defaultSegmentSize = blockDim.x * 2;
    // the last block may have smaller segment
    bool sizeDivisible = size % defaultSegmentSize == 0;
    // last block except when we only launch 1 block
    bool isLastBlock = (gridDim.x - 1 == blockIdx.x) * blockIdx.x;
    int curHalfSize = 
        isLastBlock * (ceil((size % defaultSegmentSize) / 2.0f)) + // when the last block has smaller segment
        isLastBlock * sizeDivisible * defaultHalfSize +
        !isLastBlock * defaultHalfSize;
    
    int inputIdx = threadIdx.x + blockIdx.x * defaultHalfSize * 2;
    int outputIdx = threadIdx.x + blockIdx.x * defaultHalfSize; 

    // printf("Thread %d Block %d CurHalfSize %d GridDim %d BlockDim %d\n  size %% blockDim.x = %d\n  ceil((size %% blockDim.x) / 2.0f) = %f\n",
    //        threadIdx.x, blockIdx.x, curHalfSize, gridDim.x, blockDim.x,
    //        size % blockDim.x,
    //        ceil((size % blockDim.x) / 2.0f));

    if (threadIdx.x < curHalfSize) {
        if (inputIdx + curHalfSize < size) {
            output[outputIdx] = input[inputIdx] + input[inputIdx + curHalfSize]; 
        }
        else {
            output[outputIdx] = input[inputIdx];
        }
    }
}

int main() {
    srand(time(NULL)); // Set random seed based on current time

    int N = rand() % (1024 * 8) + 10;
    float h_A[N];
    // Initialize array with random values
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
    }

    float* d_A;
    cudaMalloc((void**) &d_A, N * sizeof(float));
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);

    int halfSize = ceil(N / 2.0); 

    float* d_B;
    cudaMalloc((void**) &d_B, halfSize * sizeof(float));
    cudaMemset(d_B, 0, halfSize * sizeof(float));

    int curSize = N; 
    int numBlocks = 0;
    int numThreads = 0; 

    bool isAInput = true;
    float *inputPtr, *outputPtr; 

    while (curSize > 1) {
        numBlocks = ceil(halfSize / float(MAX_THREADS_PER_BLOCK));
        numThreads = min(halfSize, MAX_THREADS_PER_BLOCK);

        if (isAInput) {
            inputPtr = d_A; 
            outputPtr = d_B;
        } else {
            inputPtr = d_B; 
            outputPtr = d_A; 
        }

        computeSum<<<numBlocks, numThreads>>>(inputPtr, outputPtr, curSize);

        curSize = halfSize; 
        halfSize = ceil(curSize / 2.0);
        isAInput = !isAInput;
    }

    cudaDeviceSynchronize();

    float result;
    cudaMemcpy(&result, outputPtr, sizeof(float), cudaMemcpyDeviceToHost);

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

