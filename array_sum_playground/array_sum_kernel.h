#ifndef ARRAY_SUM_KERNEL_H
#define ARRAY_SUM_KERNEL_H

#include <stdio.h>
#include <cmath>

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

    if (threadIdx.x < curHalfSize) {
        if (inputIdx + curHalfSize < size) {
            output[outputIdx] = input[inputIdx] + input[inputIdx + curHalfSize]; 
        }
        else {
            output[outputIdx] = input[inputIdx];
        }
    }
}

#endif // ARRAY_SUM_KERNEL_H 