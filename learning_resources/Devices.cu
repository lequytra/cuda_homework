
void addArray(const ELEMENT *dArray, INDEX size,
	      ELEMENT *dSum);

ELEMENT multiDeviceAddArray(const ELEMENT *hA,  INDEX size)
{
  
  int numDevices;
  CU(cudaGetDeviceCount(&numDevices));
  
  INDEX numElementsPerDevice = size/numDevices;  // assume divisible
  
  // Input and output buffer for ech device and CPU
  ELEMENT **dA   = (ELEMENT **) malloc(numDevices*sizeof(ELEMENT*));
  ELEMENT **dSum = (ELEMENT **) malloc(numDevices*sizeof(ELEMENT*));
  ELEMENT  *hSum = (ELEMENT  *) malloc(numDevices*sizeof(ELEMENT));
  
  // Let each device work on its portion
  for (int dev = 0; dev < numDevices; dev++) {
    
    CU(cudaSetDevice(dev));
    
    CU(cudaMalloc((void**) &dA[dev],   sizeof(ELEMENT)*numElementsPerDevice));
    CU(cudaMalloc((void**) &dSum[dev], sizeof(ELEMENT)));
    
    CU(cudaMemcpyAsync(&dA[dev], hBuff + dev*numElementsPerDevice, cudaMemcpyHostToDevice));
    
    addArray(dA[dev], numElementsPerDevice, dSum[dev]);   // async, independent of device
  }

  
  ////////// VERSION 1
  ////////// Get sums in order and add them up on CPU
  
  for (int dev = 0; dev < numDevices; dev++) {
    
    CU(cudaSetDevice(dev));
    
    CU(cudaMemcpyAsync(&hSum[dev], dSum[dev], sizeof(ELEMENT)));
  }
  
  ELEMENT sum = 0;
  
  for (int dev = 0; dev < numDevices; dev++) {
    
    CU(cudaSetDevice(dev));
    
    CU(cudaDeviceSynchronize());
    
    sum += hSum[dev];
  }
  
  return sum;
  
  
  
  

  ////////// VERSION 2
  ////////// Get sums in order and add them up on device 0
  
  
  
  
  // Enable peer access to device 0
  CU(cudaSetDevice(0));
  
  for (int dev = 1; dev < numDevices; dev++) {
    
    CU(cudaDeviceCanAccessPeer(&canAccess, 0, dev));  // 0 can access dev's memory? 
    
    if (canAccess) {
      CU(cudaDeviceEnablePeerAccess(dev, 0));         // make dev accessible by current device
    } else {
      printf("Device 0 cannot access device %d\n", dev);
    }
  }
  
  
  // Transfer results from other devices to array dSums0 on device 0
  ELEMENT *dSums0;   
  CU(cudaMalloc((void**) &dSums0, numDevices*sizeof(ELEMENT)));
  
  CU(cudaMemcpyAsync(dSums0[0], dSum[0], cudaMemcpyDeviceToDevice));        // resukt from device 0
  
  for (int dev = 1; dev < numDevices; dev++) {
    
    CU(cudaMemcpyPeer(&dSums0[dev], 0, dSum[dev], dev, sizeof(ELEMENT)));   // results from other devices
  }
  
  addArray(dSums0, numDevices, dSum[0]);                                    // add upp all the results
  
  CU(cudaMemcpy(&hSum, dSum[0], cudaMemcpyDeviceToHost));
  
  return hSum;
  
  
  
  
  ////////// VERSION 3
  ////////// Get sums as they become available add them up on CPU
  
  
  cudaEvent_t *processed = (cudaEvent_t *) malloc(numDevices*sizeof(cudaEvent_t));
  
  
  for (int dev = 0; dev < numDevices; dev++) {
    
    CU(cudaEventCreate(&processed[dev]));
    
    CU(cudaSetDevice(dev));
    
    CU(cudaEventRecord(processed[dev]));
  }
  
  sum = 0;
  int numProcessed = 0;
  
  while (numProcessed < numDevices) {
    
    for (int dev = 0; dev < numDevices; dev++) {
      
      if (processed[dev] == NULL)                              continue; // already processed
      if (cudaEventQuery(processed[dev]) == cudaErrorNotReady) continue; // not ready
      
      CU(cudaSetDevice(dev));
      
      CU(cudaMemcpy(&hSum, dSum[dev], cudaMemcpyDeviceToHost));         // get the result to CPU
      
      sum += hSum;                                                      // process the result
      
      processed[dev] = NULL;                                            // indicate that dev processed
      
      numProcessed++;                                                   // count how many processed
    }
  }
  
  return sum;
}
