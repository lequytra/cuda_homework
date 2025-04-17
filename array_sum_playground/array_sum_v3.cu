#include <iostream>
#include "../CU.h"

using namespace std;


typedef unsigned int  ELEMENT;     // type of array elements
typedef long          INDEX;       // type of array indices


__global__ void arraySum(ELEMENT* inputPtr, ELEMENT* output) {
    INDEX halfSize = blockDim.x;

    extern __shared__ ELEMENT intermediateOutputs[];
    
    INDEX startIdx = blockIdx.x * (halfSize << 1);
    INDEX currentIndex = startIdx + threadIdx.x;
    
    // load the input and calculate intermediate sum in shared memory
    // assume input is power of 2 size.
    intermediateOutputs[threadIdx.x] = inputPtr[currentIndex] + inputPtr[currentIndex + halfSize];

    INDEX copyIdx = threadIdx.x << 1; // [0, 2, 4, 6, ...]

    for (INDEX stride =  1; stride <  halfSize; stride <<= 1) {
        __syncthreads(); 
        if (copyIdx % stride == 0 && copyIdx + stride < halfSize) {
            intermediateOutputs[copyIdx] += intermediateOutputs[copyIdx + stride];
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        output[blockIdx.x] = intermediateOutputs[0]; 
    }
}

int numElementsPerBlock() {
    cudaDeviceProp p; 
    CU(cudaGetDeviceProperties(&p, 0 /* deviceId*/)); 

    // calculate the max number of threads allowed
    int allowed = min((int)(p.sharedMemPerBlock / sizeof(ELEMENT)),
                     2*min(p.maxThreadsDim[0],
                           p.maxThreadsPerBlock));

    int results = 1; 
    
    // find the largest power of 2 within allowed
    for (results = 1; results < allowed; results <<= 1) {
    }

    if (results > allowed) {
        results >>= 1; 
    }
    return results; 
}

int main(int argc, char *argv[]) {
    // Read in the size of the array n.
    if (argc != 2) {
        cout << "ERROR:" << endl;
        cout << "Required argument: size of array" << endl;    
        cout << "Sample invocation:" << endl;
        cout << "    ./array_sum_v3 1000" << endl;        
        exit(1);
    }
        
    INDEX n = atol(argv[1]);
    INDEX N;
    // find the next power of 2 for our input size
    for (N = 1; N < n; N <<=1) {
    }

    // Initialize input array with random values
    ELEMENT *hA = (ELEMENT *) malloc(N*sizeof(ELEMENT));
    if (hA == NULL) {
        printf("Not enough memory\n");
        exit(1);
    }

    // srand(1);
    for (INDEX i = 0; i < n; i++) {
        // hA[i] = rand();
        hA[i] = 1;
    }
    for (INDEX i = n; i < N; i++) {
        hA[i] = 0; 
    }

    // Calculate grid and block dimensions
    int elementsPerBlock = min((INDEX)numElementsPerBlock(), N); //already power of 2
    
    
    int numElements = N; 
    int numThreads = elementsPerBlock / 2; 
    int numBlocks = numElements / elementsPerBlock; // both are power of 2 so should be cleanly divisible.
    bool flipped = false; 
    ELEMENT *inA, *outA;

    // Allocate and copy to device
    ELEMENT *dA, *dB;
    CU(cudaMalloc((void**)&dA, N*sizeof(ELEMENT)));
    CU(cudaMalloc((void**)&dB, numBlocks*sizeof(ELEMENT)));
    CU(cudaMemcpy(dA, hA, N*sizeof(ELEMENT), cudaMemcpyHostToDevice));

    ELEMENT result;

    for (;;) {
        int curElementsPerBlock = min(elementsPerBlock, numElements);
        numBlocks = numElements / curElementsPerBlock; // both are power of 2 so should be cleanly divisible.
        numThreads = curElementsPerBlock / 2; 

        if (flipped) {
            inA = dB;
            outA = dA;
        } else {
            inA = dA; 
            outA = dB; 
        }
        printf("Iteration: numBlocks=%d, numThreads=%d, numElements=%d, elementsPerBlock=%d, flipped=%d\n", 
               numBlocks, numThreads, numElements, curElementsPerBlock, flipped);
        // Launch kernel
        // Each threads handle 2 elements
        arraySum<<<numBlocks, numThreads, numThreads * sizeof(ELEMENT)>>>(inA, outA);

        numElements = numBlocks;
        flipped = !flipped;

        if (numElements <= 1) {
            CU(cudaDeviceSynchronize());
            CU(cudaMemcpy(&result, outA, sizeof(ELEMENT), cudaMemcpyDeviceToHost));
            break;
        }
    }    
    // Calculate expected sum on CPU for verification
    ELEMENT expected = 0;
    for (INDEX i = 0; i < N; i++) {
        expected += hA[i];
    }

    cout << "\nN: " << n << endl;
    cout << "\nSum: " << result << endl;
    cout << "Expected sum: " << expected << endl;
    cout << "Difference: " << abs((long)expected - (long)result) << endl;

    // Cleanup
    free(hA);
    cudaFree(dA);
    cudaFree(dB);
}



