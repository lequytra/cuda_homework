/*
 * Compile: nvcc hist.cu -o hist
 * Run: ./hist [options]
 * Options:
 *   --random       : Use random input values (default)
 *   --incrementing : Use incrementing input values
 *   --size N       : Set input array size (default: 1024)
 *   --bins N       : Set number of histogram bins (default: 256)
 * 
 * Example: ./hist --incrementing --size 2048 --bins 128
 */

#include <stdio.h>
#include <limits.h>
#include <stdlib.h>
#include <tuple>
#include <cuda_runtime.h>
#include <string.h>
#include "../CU.h"

typedef unsigned int    ELEMENT;
typedef long            INDEX; 

// Constants for ELEMENT type bounds
const ELEMENT ELEMENT_MAX = UINT_MAX;
const ELEMENT ELEMENT_MIN = 0;

__global__ void minmax(ELEMENT* input, ELEMENT* output, INDEX size) {
    INDEX startIdx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    INDEX t = threadIdx.x; 
    INDEX halfSize = blockDim.x; 

    // copy data to shared memory, each thread handles 2
    extern __shared__ ELEMENT mem[]; 
    ELEMENT* minIntermediate = &mem[0];
    ELEMENT* maxIntermediate = &mem[halfSize * 2];
    minIntermediate[t] = input[startIdx];
    maxIntermediate[t] = input[startIdx];
    minIntermediate[t + halfSize] = input[startIdx + halfSize];
    maxIntermediate[t + halfSize] = input[startIdx + halfSize];

    t <<= 1; 
    for (INDEX stride = 1; stride <= halfSize; stride <<= 1) {
        __syncthreads(); 
        if (t % stride == 0) {
            minIntermediate[t] = min(minIntermediate[t], minIntermediate[t + stride]);
            maxIntermediate[t] = max(maxIntermediate[t], maxIntermediate[t + stride]);
        }
    }

    __syncthreads(); // Q: Do we need __synthreads here?
    if (threadIdx.x == 0) {
        output[blockIdx.x] = minIntermediate[0];
        output[blockIdx.x + gridDim.x] = maxIntermediate[0];
    }
}

__global__ void hist(ELEMENT* input, ELEMENT* output, ELEMENT stepSize, ELEMENT numBins) {
    extern __shared__ ELEMENT copyA[];

    INDEX startIdx = blockDim.x * blockIdx.x + threadIdx.x;
    int t = threadIdx.x;

    ELEMENT* input = &copyA[0];
    ELEMENT* output = &copyA[blockDim.x];

    

}

void printUsage(const char* program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  --random       : Use random input values (default)\n");
    printf("  --incrementing : Use incrementing input values\n");
    printf("  --size N       : Set input array size (default: 1024)\n");
    printf("  --bins N       : Set number of histogram bins (default: 256)\n");
    printf("\nExample: %s --incrementing --size 2048 --bins 128\n", program_name);
}

int parseArguments(int argc, char** argv, bool* use_random, INDEX* size, INDEX* num_bins) {
    // Default parameters
    *use_random = true;
    *size = 1024;
    *num_bins = 256;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--incrementing") == 0) {
            *use_random = false;
        } else if (strcmp(argv[i], "--random") == 0) {
            *use_random = true;
        } else if (strcmp(argv[i], "--size") == 0) {
            if (i + 1 < argc) {
                *size = atoi(argv[++i]);
                if (*size <= 0) {
                    printf("Error: Size must be positive\n");
                    return 1;
                }
            } else {
                printf("Error: --size requires a value\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--bins") == 0) {
            if (i + 1 < argc) {
                *num_bins = atoi(argv[++i]);
                if (*num_bins <= 0) {
                    printf("Error: Number of bins must be positive\n");
                    return 1;
                }
            } else {
                printf("Error: --bins requires a value\n");
                return 1;
            }
        } else {
            printf("Error: Unknown option '%s'\n", argv[i]);
            return 1;
        }
    }
    return 0;
}

void printArray(const char* name, const ELEMENT* array, INDEX size, INDEX max_print = 10) {
    printf("%s: [", name);
    for (int i = 0; i < size && i < max_print; i++) {
        printf("%u", array[i]);
        if (i < size - 1 && i < max_print - 1) {
            printf(", ");
        }
    }
    if (size > max_print) {
        printf(", ...");
    }
    printf("]\n");
}

INDEX powerOf2(INDEX n) {
    INDEX N; 
    for (N = 1; N < n; N <<= 1) {}
    return N; 
}

INDEX getMaxElements() {
    cudaDeviceProp p; 
    cudaGetDeviceProperties(&p, 0); 

    INDEX numElements = min(
        // we initialize 2 arrays in shared memory
        (int)(p.sharedMemPerBlock / (2 * sizeof(ELEMENT))), 
        2 * min(
            p.maxThreadsPerBlock,
            p.maxThreadsDim[0] // 1 thread handles 2 elments
        )
    );
    return powerOf2(numElements); 
}

void verifyMinMaxResults(ELEMENT gpuMin, ELEMENT gpuMax, ELEMENT cpuMin, ELEMENT cpuMax) {
    printf("\nVerifying results:\n");
    printf("GPU Min: %u, Max: %u\n", gpuMin, gpuMax);
    printf("CPU Min: %u, Max: %u\n", cpuMin, cpuMax);
    
    if (gpuMin == cpuMin && gpuMax == cpuMax) {
        printf("Results match! Verification successful.\n");
    } else {
        printf("ERROR: Results do not match!\n");
        if (gpuMin != cpuMin) printf("Min values differ - GPU: %u, CPU: %u\n", gpuMin, cpuMin);
        if (gpuMax != cpuMax) printf("Max values differ - GPU: %u, CPU: %u\n", gpuMax, cpuMax);
    }
}

int main(int argc, char** argv) {
    // Parameters to be set by parseArguments
    bool use_random;
    INDEX size;
    INDEX num_bins;
    
    // Parse command line arguments
    if (parseArguments(argc, argv, &use_random, &size, &num_bins) != 0) {
        printUsage(argv[0]);
        return 1;
    }

    INDEX sizePadded = powerOf2(size); 
    
    printf("Running with parameters:\n");
    printf("  Input type: %s\n", use_random ? "random" : "incrementing");
    printf("  Array size: %ld, padded: %ld\n", size, sizePadded);
    printf("  Number of bins: %ld\n", num_bins);
    
    // Host memory allocation
    ELEMENT* hInput = (ELEMENT*)malloc(sizePadded * sizeof(ELEMENT));
    
    // Initialize input array
    if (use_random) {
        printf("Using random input values\n");
        for (int i = 0; i < size; i++) {
            hInput[i] = rand();  // Values between 0 and num_bins-1
        }
    } else {
        printf("Using incrementing input values\n");
        for (int i = 0; i < size; i++) {
            hInput[i] = i;  // Values cycle from 0 to num_bins-1
        }
    }
    // pad with the last elements
    for (INDEX i = size; i < sizePadded; i++) {
        hInput[i] = hInput[i -  1]; 
    }
    
    // Print first few elements of input array
    printArray("Input array", hInput, size);
    
    INDEX maxElements = min(getMaxElements(), sizePadded);
    INDEX numElements = sizePadded; 
    int numBlocks = ceil(numElements / maxElements); 
    int numThreads = min(numElements, maxElements); 

    // Device memory allocation
    ELEMENT* dInput;
    CU(cudaMalloc((void**)&dInput, sizePadded * sizeof(ELEMENT)));
    // Copy input data to device
    CU(cudaMemcpy(dInput, hInput, sizePadded * sizeof(ELEMENT), cudaMemcpyHostToDevice));
    

    ELEMENT* dOutput;
    
    CU(cudaMalloc((void**)&dOutput, numBlocks * 2 * sizeof(ELEMENT)));
    INDEX curSize = size; 
    ELEMENT hOut[2];
    // Launch kernel
    for (;;) {
        numBlocks = ceil((float)numElements / maxElements);
        INDEX curNumElements = min(numElements, maxElements);
        numThreads = curNumElements / 2; 
        printf("  Blocks: %d, Threads: %d\n", numBlocks, numThreads);
        // TODO: Need to properly swapped out input and output here per iteration
        minmax<<<numBlocks, numThreads, curNumElements * 2 * sizeof(ELEMENT)>>>(dInput, dOutput, curSize); 

        numElements = numBlocks * 2; 
        curSize = numElements; 

        if (numBlocks <= 1) {
            CU(cudaDeviceSynchronize());
            CU(cudaMemcpy(&hOut, dOutput, 2 * sizeof(ELEMENT), cudaMemcpyDeviceToHost));
            break;
        }

        ELEMENT* temp; 
        temp = dInput; 
        dInput = dOutput; 
        dOutput = temp; 
    }
    
    // CPU verification
    ELEMENT cpuMin = ELEMENT_MAX;
    ELEMENT cpuMax = ELEMENT_MIN;
    
    // Simple sequential min/max calculation for verification
    for (INDEX i = 0; i < size; i++) {
        cpuMin = min(cpuMin, hInput[i]);
        cpuMax = max(cpuMax, hInput[i]);
    }
    
    // Compare GPU and CPU results
    verifyMinMaxResults(hOut[0], hOut[1], cpuMin, cpuMax);

    // Free host memory
    free(hInput);
    
    // // Free device memory
    CU(cudaFree(dInput));
    CU(cudaFree(dOutput));
    
    printf("Program completed successfully!\n");
    return 0;
}
