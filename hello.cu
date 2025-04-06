#include <stdio.h>

// nvcc -o hello hello.cu && ./hello

__global__ void cuda_hello() {
    // threads and blocks are identified by 3D box
    // this is historical for graphic
    printf("Hello World from GPU! %d, %d, %d, %d, %d, %d\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
}

int main() {
    cuda_hello<<<3, 5>>>();
    cudaDeviceSynchronize();
    
    return 0;
}
