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

// ANSI color codes
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_RESET   "\x1b[0m"

typedef unsigned int    ELEMENT;
typedef long            INDEX; 

// Constants for ELEMENT type bounds
const ELEMENT ELEMENT_MAX = UINT_MAX;
const ELEMENT ELEMENT_MIN = 0;

const int MAX_NUM_STREAMS = 4; 

template <typename T, typename IndexType>
__device__ T get(T* A, IndexType i, IndexType j, IndexType N) {
    return A[i * N + j];
}

template <typename T, typename IndexType>
__device__ void set(T* A, IndexType i, IndexType j, IndexType N, T value) {
    A[i * N + j] = value;
}

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

__global__ void hist(
    ELEMENT* input,
    INDEX* output,
    ELEMENT stepSize,
    INDEX numBins,
    ELEMENT min,
    INDEX size
) {
    INDEX startIdx = blockDim.x * blockIdx.x + threadIdx.x;
    int t = threadIdx.x;

    extern __shared__ INDEX histRes[];

    if (t < numBins) {
        histRes[t] = 0; 
    }
    __syncthreads();

    if (startIdx < size) {
        ELEMENT diff = input[startIdx] - min;
        INDEX curHist;
        if (stepSize <= 0) {
            curHist = 0; // Handle invalid stepSize
        } else if (diff > ELEMENT_MAX - (stepSize - 1)) {
            curHist = diff / stepSize + (diff % stepSize != 0); // Safe ceiling
        } else {
            curHist = (diff + stepSize - 1) / stepSize;
        }
        atomicAdd((unsigned long long*)&histRes[curHist], (unsigned long long) 1);
    }

    __syncthreads();
    if (t < numBins) {
        set<INDEX, INDEX>(output, t, blockIdx.x, gridDim.x, histRes[t]);
        // output[blockIdx.x * numBins + t] = histRes[t];
    }

}

__global__ void addKernel(INDEX inNum, const INDEX *inA, INDEX *outA)
{
  
  INDEX numElementsPerBlock = blockDim.x << 1;         // numElementsPerBlock is twice the number of threads per block
  INDEX numElementsHalf     = blockDim.x;
  
  INDEX inStart  = blockIdx.x * numElementsPerBlock;   // Portion of inA that this block works on
  INDEX outStart = blockIdx.x;                         // Where in outA to place the result of this block

  int t = threadIdx.x;

  // STAGE 1: Copy our portion of inA into a shared memory array copyA,
  // so that the summing up can be done using shared memory.
  // If our portion of inA is smaller than numElementsPerBlock,
  // then fill the extra elements of copyA with 0.
  // (Adding extra 0's has no effect on the sum.)
  
  extern __shared__ ELEMENT copyA[];

  // Each thread copies 2 elements, separated by numElementsHalf
  
  INDEX inIndex   = t + inStart;    // which element of inA to copy
  INDEX copyIndex = t;              // where in copyA to copy it
  
  copyA[copyIndex]                = inIndex < inNum ? inA[inIndex] : 0;
  inIndex   += numElementsHalf;
  copyIndex += numElementsHalf;
  copyA[copyIndex]                = inIndex < inNum ? inA[inIndex] : 0;

  

  // STAGE 2: Percolate sum to copyA[0].
  
  copyIndex = t<<1;       // which index in copyA this thread t assigns

  // This thread t adds to its element of copyA another element,
  // farther away by the amount stride.
  // The stride keep growing by a factor of 2 on each itearation
  
  for (int stride = 1; stride < numElementsPerBlock; stride <<= 1) {
    __syncthreads();   // wait for all assignments of copyA to finish
    if (t % stride == 0) {
      copyA[copyIndex] += copyA[copyIndex + stride];
    }
  }

  
  // The sum is in position 0
  // No need to sychronize threads because last iteration was performed only by thread 0
  if (t == 0) {
    outA[outStart] = copyA[0];
  }
}

void printUsage(const char* program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  --random       : Use random input values (default)\n");
    printf("  --incrementing : Use incrementing input values\n");
    printf("  --size N       : Set input array size (default: 1024)\n");
    printf("  --bins N       : Set number of histogram bins (default: 256)\n");
    printf("  --verify       : Enable verification of minmax results (default: false)\n");
    printf("\nExample: %s --incrementing --size 2048 --bins 128\n", program_name);
}

int parseArguments(int argc, char** argv, bool* use_random, INDEX* size, INDEX* numBins, bool* verify) {
    // Default parameters
    *use_random = true;
    *size = 1024;
    *numBins = 256;
    *verify = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--incrementing") == 0) {
            *use_random = false;
        } else if (strcmp(argv[i], "--random") == 0) {
            *use_random = true;
        } else if (strcmp(argv[i], "--verify") == 0) {
            *verify = true;
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
                *numBins = atoi(argv[++i]);
                if (*numBins <= 0) {
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
    INDEX powerOf2Element = powerOf2(numElements);
    if (powerOf2Element > numElements) {
        powerOf2Element >>= 1; 
    }
    return powerOf2Element;
}

void verifyMinMaxResults(ELEMENT gpuMin, ELEMENT gpuMax, ELEMENT cpuMin, ELEMENT cpuMax) {
    printf("\nVerifying results:\n");
    printf("GPU Min: %u, Max: %u\n", gpuMin, gpuMax);
    printf("CPU Min: %u, Max: %u\n", cpuMin, cpuMax);
    
    if (gpuMin == cpuMin && gpuMax == cpuMax) {
        printf(ANSI_COLOR_GREEN "Results match! Verification successful." ANSI_COLOR_RESET "\n");
    } else {
        printf(ANSI_COLOR_RED "ERROR: Results do not match!\n");
        if (gpuMin != cpuMin) printf("Min values differ - GPU: %u, CPU: %u\n", gpuMin, cpuMin);
        if (gpuMax != cpuMax) printf("Max values differ - GPU: %u, CPU: %u\n", gpuMax, cpuMax);
        printf(ANSI_COLOR_RESET);
    }
}

void verifyHistResults(
    INDEX* dLocalHists,
    ELEMENT* hInput,
    INDEX numBlocks,
    INDEX numBins,
    INDEX size,
    ELEMENT stepSize,
    ELEMENT minVal,
    bool needComputeFinal
) {
    // Allocate and compute final histogram on CPU
    INDEX *hFinalHist, *hLocalHists;
    if (needComputeFinal) {
        hFinalHist = (INDEX*)calloc(numBins, sizeof(INDEX));
        hLocalHists = (INDEX*)malloc(numBlocks * numBins * sizeof(INDEX));
        // [numBins, numBlocks]
        CU(cudaMemcpy(hLocalHists, dLocalHists, numBins * numBlocks * sizeof(INDEX), cudaMemcpyDeviceToHost));
        // Sum up local histograms from each block
        for (INDEX block = 0; block < numBlocks; block++) {
            for (INDEX bin = 0; bin < numBins; bin++) {
                hFinalHist[bin] += hLocalHists[bin * numBlocks + block];
                // hFinalHist[bin] += hLocalHists[block * numBins + bin];
            }
        }
    } 
    else {
        hFinalHist = dLocalHists; 
    }
    
    // Compute CPU histogram for verification
    INDEX* hCPUHist = (INDEX*)calloc(numBins, sizeof(INDEX));
    for (INDEX i = 0; i < size; i++) {
        ELEMENT diff = hInput[i] - minVal;
        INDEX binIdx;
        if (stepSize <= 0) {
            binIdx = 0;
        } else if (diff > ELEMENT_MAX - (stepSize - 1)) {
            binIdx = diff / stepSize + (diff % stepSize != 0);
        } else {
            binIdx = (diff + stepSize - 1) / stepSize;
        }

        if (binIdx < numBins) {
            hCPUHist[binIdx]++;
        }
    }

    // Verify results
    bool mismatch = false;
    printf("\nHistogram Verification:\n");
    for (INDEX i = 0; i < numBins; i++) {
        if (hFinalHist[i] != hCPUHist[i]) {
            printf(ANSI_COLOR_RED "Mismatch at bin %ld: GPU = %ld, CPU = %ld\n" ANSI_COLOR_RESET, 
                   i, hFinalHist[i], hCPUHist[i]);
            mismatch = true;
        }
    }
    
    if (!mismatch) {
        printf(ANSI_COLOR_GREEN "Histogram verification successful! All bins match.\n" ANSI_COLOR_RESET);
    }

    // Print first few bins of the histogram
    printf("\nFirst 10 histogram bins:\n");
    printf("Bin:  ");
    for (INDEX i = 0; i < min(numBins, (INDEX)10); i++) {
        printf("%8ld ", i);
    }
    printf("\nCount:");
    for (INDEX i = 0; i < min(numBins, (INDEX)10); i++) {
        printf("%8ld ", hFinalHist[i]);
    }
    printf("\n");

    // Cleanup
    if (needComputeFinal) {
        free(hLocalHists);
        free(hFinalHist);
    }
    free(hCPUHist);
}

INDEX getMaxElementsAdd() {
    cudaDeviceProp p; 
    cudaGetDeviceProperties(&p, 0); 

    INDEX numElements = min(
        // we initialize 2 arrays in shared memory
        (int)(p.sharedMemPerBlock / sizeof(INDEX)), 
        2 * min(
            p.maxThreadsPerBlock,
            p.maxThreadsDim[0] // 1 thread handles 2 elments
        )
    );
    INDEX powerOf2Element = powerOf2(numElements);
    if (powerOf2Element > numElements) {
        powerOf2Element >>= 1; 
    }
    return powerOf2Element;
}

int main(int argc, char** argv) {
    // Parameters to be set by parseArguments
    bool use_random;
    INDEX size;
    INDEX numBins;
    bool verify;
    
    // Parse command line arguments
    if (parseArguments(argc, argv, &use_random, &size, &numBins, &verify) != 0) {
        printUsage(argv[0]);
        return 1;
    }

    INDEX sizePadded = powerOf2(size); 
    
    printf("Running with parameters:\n");
    printf("  Input type: %s\n", use_random ? "random" : "incrementing");
    printf("  Array size: %ld, padded: %ld\n", size, sizePadded);
    printf("  Number of bins: %ld\n", numBins);
    printf("  Verification: %s\n", verify ? "enabled" : "disabled");
    
    // Host memory allocation
    ELEMENT* hInput = (ELEMENT*)malloc(sizePadded * sizeof(ELEMENT));
    
    // Initialize input array
    if (use_random) {
        printf("Using random input values\n");
        for (int i = 0; i < size; i++) {
            hInput[i] = rand();  // Values between 0 and numBins-1
        }
    } else {
        printf("Using incrementing input values\n");
        for (int i = 0; i < size; i++) {
            hInput[i] = i;  // Values cycle from 0 to numBins-1
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
    INDEX numBlocks = ceil(numElements / maxElements); 
    int numThreads = min(numElements, maxElements); 

    // Device memory allocation
    ELEMENT* dInput;
    ELEMENT* dInputCopy;
    CU(cudaMalloc((void**)&dInput, sizePadded * sizeof(ELEMENT)));
    CU(cudaMalloc((void**)&dInputCopy, sizePadded * sizeof(ELEMENT)));
    // Copy input data to device
    CU(cudaMemcpy(dInput, hInput, sizePadded * sizeof(ELEMENT), cudaMemcpyHostToDevice));
    // Make a copy of the input array
    CU(cudaMemcpy(dInputCopy, dInput, sizePadded * sizeof(ELEMENT), cudaMemcpyDeviceToDevice));
    

    ELEMENT* dOutput;
    
    CU(cudaMalloc((void**)&dOutput, numBlocks * 2 * sizeof(ELEMENT)));
    INDEX curSize = size; 
    ELEMENT hOut[2];
    // Launch kernel
    for (;;) {
        numBlocks = ceil((float)numElements / maxElements);
        INDEX curNumElements = min(numElements, maxElements);
        numThreads = curNumElements / 2; 

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
    
    if (verify) {
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
    } else {
        printf("\nResults:\n");
        printf("GPU Min: %u, Max: %u\n", hOut[0], hOut[1]);
    }
    ELEMENT diff = hOut[1] - hOut[0];
    ELEMENT stepSize;
    if (diff > ELEMENT_MAX - numBins) {
        stepSize = diff / numBins + (diff % numBins != 0); // Add 1 if there's a remainder
    } else {
        stepSize = (diff + numBins - 1) / numBins;
    }

    // each thread now handle only a single element
    maxElements = maxElements / 2; 
    numBlocks = ceil(sizePadded / (float)maxElements); 
    numThreads = min(sizePadded, maxElements); 

    INDEX* dLocalHists;
    CU(cudaMalloc(&dLocalHists, numBins * numBlocks * sizeof(INDEX)));

    hist<<<numBlocks, numThreads, numBins * sizeof(INDEX)>>>(
        dInputCopy,
        dLocalHists,
        stepSize, 
        numBins, 
        hOut[0],
        size
    );

    if (verify) {
        CU(cudaDeviceSynchronize());
        // Verify histogram results
        verifyHistResults(dLocalHists, hInput, numBlocks, numBins, size, stepSize, hOut[0], true);
    }

    int numStreams = min(numBins, (INDEX) MAX_NUM_STREAMS);
    cudaStream_t streams[numStreams];
    
    for (int s = 0; s < numStreams; s++) {
        cudaStreamCreate(&streams[s]);
    }
    INDEX maxElementsAdd = getMaxElementsAdd();
    numElements = powerOf2(numBlocks); 

    INDEX* hHistOutput = (INDEX*)malloc(numBins * sizeof(INDEX));

    for (INDEX curBin = 0; curBin < numBins; curBin++) {
        INDEX elementsPerBlock = min(maxElementsAdd, numElements);
        INDEX numAddBlocks = ceil(elementsPerBlock / (float)maxElementsAdd);
        curSize = numBlocks; 

        INDEX *dAddIn = dLocalHists + curBin * numBlocks;
        INDEX *dAddOut; 
        CU(cudaMalloc(&dAddOut, numAddBlocks * sizeof(INDEX)));

        for (;;) {
            printf("Bin %ld: blocks=%ld, threads=%ld, elements per block=%ld, current size=%ld\n",
                   curBin, numAddBlocks, elementsPerBlock >> 2, elementsPerBlock, curSize);
            addKernel<<<
                numAddBlocks, 
                elementsPerBlock >> 1, 
                elementsPerBlock * sizeof(INDEX), 
                streams[curBin % numStreams]
            >>>(
                curSize,
                dAddIn,
                dAddOut
            );

            if (numAddBlocks <= 1) {
                CU(cudaMemcpyAsync(
                    hHistOutput + curBin,
                    dAddOut,
                    sizeof(INDEX),
                    cudaMemcpyDeviceToHost
                ));
                break;
            }

            elementsPerBlock = numAddBlocks; 
            curSize = numAddBlocks;
            numAddBlocks = ceil(elementsPerBlock / (float)maxElementsAdd);

            INDEX* temp = dAddIn;
            dAddIn = dAddOut;
            dAddOut = temp;
        }
    }
    
    if (verify) {
        CU(cudaDeviceSynchronize());
        // Verify histogram results
        verifyHistResults(hHistOutput, hInput, numBlocks, numBins, size, stepSize, hOut[0], false);
    }
    

    for (INDEX s = 0; s < numStreams; s++) {
        CU(cudaStreamDestroy(streams[s]));
    }
    // Free host memory
    free(hInput);
    
    // // Free device memory
    CU(cudaFree(dInput));
    CU(cudaFree(dOutput));
    CU(cudaFree(dLocalHists));
    
    printf(ANSI_COLOR_GREEN "Program completed successfully!\n" ANSI_COLOR_RESET);
    return 0;
}
