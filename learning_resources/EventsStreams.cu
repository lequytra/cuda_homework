

__host__            cudaError_t cudaEventCreate      ( cudaEvent_t* event );
__host__ __device__ cudaError_t cudaEventDestroy     ( cudaEvent_t  event );
__host__            cudaError_t cudaEventElapsedTime ( float* ms, cudaEvent_t start, cudaEvent_t end );
__host__            cudaError_t cudaEventQuery       ( cudaEvent_t event );
__host__ __device__ cudaError_t cudaEventRecord      ( cudaEvent_t event, cudaStream_t stream = 0) ;
__host__            cudaError_t cudaEventSynchronize ( cudaEvent_t event);
__host__            cudaError_t cudaStreamCreate     ( cudaStream_t*stream );
__host__ __device__ cudaError_t cudaStreamDestroy    ( cudaStream_t stream );
__host__ __device__ cudaError_t cudaStreamWaitEvent  ( cudaStream_t stream, cudaEvent_t event, unsigned int  flags = 0 );


int f()
{
  int numBlocks, numThreads, numSharedBytes, size;
  
  int *hBuff1, *hBuff2, *dBuff1, *dBuff2;

  float time;
    
  cudaStream_t transferStream, workStream;

  cudaStreamCreate(&transferStream);
  cudaStreamCreate(&workStream);
    
  cudaEvent_t empty1, empty2, full1, full2;

  cudaEventCreate(&empty1);
  cudaEventCreate(&empty2);
  cudaEventCreate(&full1);
  cudaEventCreate(&full2);            
    
  cudaEventRecord(empty1, workStream);    
  cudaEventRecord(empty2, workStream);

    
  for (;;) {
    cudaEventSynchronize(empty1);
    cudaMemcpyAsync(dBuff1, hBuff1, size, cudaMemcpyHostToDevice, transferStream);
    cudaEventRecord(full1, transferStream);
      
    cudaEventSynchronize(empty2);
    cudaMemcpyAsync(dBuff2, hBuff2, size, cudaMemcpyHostToDevice, transferStream);
    cudaEventRecord(full2, transferStream);
      
    cudaEventSynchronize(full1);
    kernel<<<numBlocks, numThreads, numSharedBytes, workStream>>>(dBuff1);
    cudaEventRecord(empty1, workStream);
      
    cudaEventSynchronize(full2);
    kernel<<<numBlocks, numThreads, numSharedBytes, workStream>>>(dBuff2);
    cudaEventRecord(empty2, workStream);

    /*
      cudaEventElapsedTime(&time, empty2, full2);
      printf("Data transfer took: %f ms\n", time);
    */
  }
    
  cudaStreamDestroy(transferStream);
  cudaStreamDestroy(workStream);
    
  cudaEventDestroy(empty1);
  cudaEventDestroy(empty2);
  cudaEventDestroy(full1);
  cudaEventDestroy(full2);            
}