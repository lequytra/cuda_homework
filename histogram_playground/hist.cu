#include <stdio.h>
#include <stdlib.h>
#include <tuple>
#include <cuda_runtime.h>
#include <string.h>
#include "../CU.h"

typedef unsigned int    ELEMENT;
typedef long            INDEX; 

template <typename Op>
__global__ void reduce(ELEMENT* input, ELEMENT* output, Op op) {
    INDEX startIdx = blockIdx.x * blockDim.x + threadIdx.x;
    INDEX t = threadIdx.x; 
    INDEX halfSize = blockDim.x / 2; 

    // copy data to shared memory, each thread handles 2
    extern __shared__ ELEMENT intermediate[]; 
    intermediate[t] = input[startIdx];
    intermediate[t + halfSize] = input[startIdx + halfSize];

    t <<= 1; 
    for (INDEX stride = 1; stride < halfSize; stride <<= 1) {
        __syncthreads(); 
        if (t % stride == 0) {
            intermediate[t] = op(intermediate[t], intermediate[t + stride]);
        }
    }

    __syncthreads(); // Q: Do we need __synthreads here?
    if (threadIdx.x == 0) {
        output[blockIdx.x] = intermediate[0];
    }

}

void print_usage(const char* program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  --random       : Use random input values (default)\n");
    printf("  --incrementing : Use incrementing input values\n");
    printf("  --size N       : Set input array size (default: 1024)\n");
    printf("  --bins N       : Set number of histogram bins (default: 256)\n");
    printf("\nExample: %s --incrementing --size 2048 --bins 128\n", program_name);
}

int parse_arguments(int argc, char** argv, bool* use_random, int* size, int* num_bins) {
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

void print_array(const char* name, const unsigned int* array, int size, int max_print = 10) {
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

int main(int argc, char** argv) {
    // Parameters to be set by parse_arguments
    bool use_random;
    int size;
    int num_bins;
    
    // Parse command line arguments
    if (parse_arguments(argc, argv, &use_random, &size, &num_bins) != 0) {
        print_usage(argv[0]);
        return 1;
    }
    
    printf("Running with parameters:\n");
    printf("  Input type: %s\n", use_random ? "random" : "incrementing");
    printf("  Array size: %d\n", size);
    printf("  Number of bins: %d\n", num_bins);
    
    // Host memory allocation
    unsigned int* h_input = (unsigned int*)malloc(size * sizeof(unsigned int));
    unsigned int* h_output = (unsigned int*)malloc(num_bins * sizeof(unsigned int));
    
    // Initialize input array
    if (use_random) {
        printf("Using random input values\n");
        for (int i = 0; i < size; i++) {
            h_input[i] = rand() % num_bins;  // Values between 0 and num_bins-1
        }
    } else {
        printf("Using incrementing input values\n");
        for (int i = 0; i < size; i++) {
            h_input[i] = i % num_bins;  // Values cycle from 0 to num_bins-1
        }
    }
    
    // Print first few elements of input array
    print_array("Input array", h_input, size);
    
    // Device memory allocation
    unsigned int* d_input;
    unsigned int* d_output;
    CU(cudaMalloc((void**)&d_input, size * sizeof(unsigned int)));
    CU(cudaMalloc((void**)&d_output, num_bins * sizeof(unsigned int)));
    
    // Copy input data to device
    CU(cudaMemcpy(d_input, h_input, size * sizeof(unsigned int), cudaMemcpyHostToDevice));
    
    // Initialize output array to zero
    CU(cudaMemset(d_output, 0, num_bins * sizeof(unsigned int)));
    
    // Launch kernel
    // TODO: Configure grid and block dimensions
    // histogram_kernel<<<grid, block>>>(d_input, d_output, size);
    
    // Copy result back to host
    CU(cudaMemcpy(h_output, d_output, num_bins * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    
    // Print output array (should be all zeros since kernel is not implemented)
    print_array("Output array", h_output, num_bins);
    
    // Free device memory
    CU(cudaFree(d_input));
    CU(cudaFree(d_output));
    
    // Free host memory
    free(h_input);
    free(h_output);
    
    printf("Program completed successfully!\n");
    return 0;
}
