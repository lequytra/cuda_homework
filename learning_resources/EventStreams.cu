

void CPUwaits()
{
  int numBlocks, numThreads, numSharedBytes, size;
  
  int *hBuff1, *hBuff2, *dBuff1, *dBuff2;

  float time;
    
  cudaStream_t transferStream, workStream;

  CU(cudaStreamCreate(&transferStream));
  CU(cudaStreamCreate(&workStream));
    
  cudaEvent_t empty1, empty2, full1, full2;

  CU(cudaEventCreate(&empty1));
  CU(cudaEventCreate(&empty2));
  CU(cudaEventCreate(&full1));
  CU(cudaEventCreate(&full2));            
    
  CU(cudaEventRecord(empty1, workStream));    
  CU(cudaEventRecord(empty2, workStream));

    
  for (;;) {
    CU(cudaEventSynchronize(empty1));
    // wait for hBuff1 to be filled
    CU(cudaMemcpyAsync(dBuff1, hBuff1, size, cudaMemcpyHostToDevice, transferStream));
    CU(cudaEventRecord(full1, transferStream));
      
    CU(cudaEventSynchronize(empty2));
    // wait for hBuff2 to be filled
    CU(cudaMemcpyAsync(dBuff2, hBuff2, size, cudaMemcpyHostToDevice, transferStream));
    CU(cudaEventRecord(full2, transferStream));
      
    CU(cudaEventSynchronize(full1));
    kernel<<<numBlocks, numThreads, numSharedBytes, workStream>>>(dBuff1);
    CU(cudaEventRecord(empty1, workStream));
      
    CU(cudaEventSynchronize(full2));
    kernel<<<numBlocks, numThreads, numSharedBytes, workStream>>>(dBuff2);
    CU(cudaEventRecord(empty2, workStream));

    /*
      CU(cudaEventElapsedTime(&time, empty2, full2));
      printf("Data transfer took: %f ms\n", time);
    */
  }
    
  CU(cudaStreamDestroy(transferStream));
  CU(cudaStreamDestroy(workStream));
    
  CU(cudaEventDestroy(empty1));
  CU(cudaEventDestroy(empty2));
  CU(cudaEventDestroy(full1));
  CU(cudaEventDestroy(full2));            
}







void GPUwaits()
{
  int numBlocks, numThreads, numSharedBytes, size;
  
  int *hBuff1, *hBuff2, *dBuff1, *dBuff2;

  float time;
    
  cudaStream_t transferStream, workStream;

  CU(cudaStreamCreate(&transferStream));
  CU(cudaStreamCreate(&workStream));
    
  cudaEvent_t empty1, empty2, full1, full2;

  CU(cudaEventCreate(&empty1));
  CU(cudaEventCreate(&empty2));
  CU(cudaEventCreate(&full1));
  CU(cudaEventCreate(&full2));            
    
  CU(cudaEventRecord(empty1, workStream));    
  CU(cudaEventRecord(empty2, workStream));


  // host data already ready
  for (int i = 0; i < numPreparedData; i++) {
    
    CU(cudaStreamWaitEvent(transferStream, empty1));
    CU(cudaMemcpyAsync(dBuff1, hBuff1[i], size, cudaMemcpyHostToDevice, transferStream));
    CU(cudaEventRecord(full1, transferStream));
      
    CU(cudaStreamWaitEvent(transferStream, empty2));
    CU(cudaMemcpyAsync(dBuff2, hBuff2[i], size, cudaMemcpyHostToDevice, transferStream));
    CU(cudaEventRecord(full2, transferStream));
      
    CU(cudaStreamWaitEvent(workStream, full1));
    kernel<<<numBlocks, numThreads, numSharedBytes, workStream>>>(dBuff1);
    CU(cudaEventRecord(empty1, workStream));
      
    CU(cudaStreamWaitEvent(workStream, full2));
    kernel<<<numBlocks, numThreads, numSharedBytes, workStream>>>(dBuff2);
    CU(cudaEventRecord(empty2, workStream));

  }
    
  CU(cudaStreamDestroy(transferStream));
  CU(cudaStreamDestroy(workStream));
    
  CU(cudaEventDestroy(empty1));
  CU(cudaEventDestroy(empty2));
  CU(cudaEventDestroy(full1));
  CU(cudaEventDestroy(full2));            
}
