
__host__ __device__ cudaError_t cudaFree ( void* devPtr );
__host__            cudaError_t cudaFreeHost ( void* ptr );
__host__ __device__ cudaError_t cudaMalloc ( void** devPtr, size_t size );
__host__            cudaError_t cudaMallocHost ( void** ptr, size_t size );
__host__            cudaError_t cudaMallocManaged ( void** devPtr, size_t size, unsigned int flags = cudaMemAttachGlobal);
__host__            cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind );
__host__ __device__ cudaError_t cudaMemcpyAsync ( void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0 );
__host__            cudaError_t cudaMemset ( void* devPtr, int  value, size_t count );
__host__ __device__ cudaError_t cudaMemsetAsync ( void* devPtr, int  value, size_t count, cudaStream_t stream = 0 );


void f()
{
  int size = 100;
  int numBlocks, numThreads;
  
  {// Basic allocation cudaMalloc

    int *hBuff = (int *) malloc(size);
    hBuff[0] = 42;
  
    int *dBuff;
    cudaMalloc(&dBuff, size);
    cudaMemcpy(dBuff, hBuff, size, cudaMemcpyHostToDevice);

    kernel<<<numBlocks, numThreads>>>(dBuff);
    
    cudaFree(dBuff);
  }

  {// Allocation in pinned memory cudaMallocHost
    
    int *hBuff = (int *) malloc(size);
    hBuff[0] = 42;
    
    int *dBuff;
    cudaMallocHost(&dBuff, size);
    cudaMemcpy(dBuff, hBuff, size, cudaMemcpyHostToDevice);
    
    kernel<<<numBlocks, numThreads>>>(dBuff);
  
    cudaFreeHost(dBuff);
  }

    
  {// Unified Memory Model
    
    int *dBuff;
    cudaMallocManaged(&dBuff, size);
    dBuff[0] = 42;
    
    kernel<<<numBlocks, numThreads>>>(dBuff);
    
    cudaFree(dBuff);
  }
}