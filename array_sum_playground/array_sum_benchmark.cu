#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <chrono>
#include "array_sum_kernel.h"

// nvcc -o array_sum_benchmark array_sum_benchmark.cu && ./array_sum_benchmark

// CPU implementation for array summation
float cpuSum(float* input, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += input[i];
    }
    return sum;
}

// GPU implementation for array summation with timing
float gpuSum(float* h_input, int size, double* memcpy_time = nullptr, double* compute_time = nullptr) {
    float* d_A;
    cudaMalloc((void**) &d_A, size * sizeof(float));
    
    // Measure memory transfer time
    auto memcpy_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_A, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    auto memcpy_end = std::chrono::high_resolution_clock::now();
    if (memcpy_time) {
        *memcpy_time = std::chrono::duration<double, std::milli>(memcpy_end - memcpy_start).count();
    }

    int halfSize = ceil(size / 2.0); 

    float* d_B;
    cudaMalloc((void**) &d_B, halfSize * sizeof(float));
    cudaMemset(d_B, 0, halfSize * sizeof(float));

    int curSize = size; 
    int numBlocks = 0;
    int numThreads = 0; 

    bool isAInput = true;
    float *inputPtr, *outputPtr; 

    // Measure computation time
    auto compute_start = std::chrono::high_resolution_clock::now();
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
    auto compute_end = std::chrono::high_resolution_clock::now();
    if (compute_time) {
        *compute_time = std::chrono::duration<double, std::milli>(compute_end - compute_start).count();
    }

    float result;
    cudaMemcpy(&result, outputPtr, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    
    return result;
}

// Benchmark function
void benchmark(int size, int numRuns) {
    // Allocate and initialize input array
    float* h_input = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        h_input[i] = (float)rand() / RAND_MAX;
    }
    
    // CPU benchmark
    auto cpu_start = std::chrono::high_resolution_clock::now();
    float cpu_result = 0.0f;
    for (int i = 0; i < numRuns; i++) {
        cpu_result = cpuSum(h_input, size);
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_elapsed = cpu_end - cpu_start;
    
    // GPU benchmark with detailed timing
    double total_memcpy_time = 0.0;
    double total_compute_time = 0.0;
    float gpu_result = 0.0f;
    
    for (int i = 0; i < numRuns; i++) {
        double memcpy_time, compute_time;
        gpu_result = gpuSum(h_input, size, &memcpy_time, &compute_time);
        total_memcpy_time += memcpy_time;
        total_compute_time += compute_time;
    }
    
    // Print results
    printf("Array size: %d\n", size);
    printf("CPU sum: %.6f, Time: %.2f ms (avg: %.2f ms per run)\n", 
           cpu_result, cpu_elapsed.count(), cpu_elapsed.count() / numRuns);
    printf("GPU sum: %.6f\n", gpu_result);
    printf("GPU Memory Transfer Time: %.2f ms (avg: %.2f ms per run)\n", 
           total_memcpy_time, total_memcpy_time / numRuns);
    printf("GPU Computation Time: %.2f ms (avg: %.2f ms per run)\n", 
           total_compute_time, total_compute_time / numRuns);
    printf("GPU Total Time: %.2f ms (avg: %.2f ms per run)\n", 
           total_memcpy_time + total_compute_time, 
           (total_memcpy_time + total_compute_time) / numRuns);
    printf("Speedup (excluding memory transfer): %.2fx\n", 
           cpu_elapsed.count() / total_compute_time);
    printf("Speedup (including memory transfer): %.2fx\n", 
           cpu_elapsed.count() / (total_memcpy_time + total_compute_time));
    printf("Difference: %.6f\n\n", fabs(cpu_result - gpu_result));
    
    free(h_input);
}

int main() {
    srand(time(NULL)); // Set random seed based on current time
    
    // Test with different array sizes
    const int sizes[] = {1000, 10000, 100000, 1000000, 10000000};
    const int numRuns = 10; // Number of runs for each size
    
    printf("Benchmarking GPU vs CPU array summation\n");
    printf("======================================\n\n");
    
    for (int i = 0; i < 5; i++) {
        benchmark(sizes[i], numRuns);
    }
    
    return 0;
} 