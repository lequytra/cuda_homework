
#include <iostream>
#include "cuda.h"
#include "../../CU.h"
#include "Timer.h"

using namespace std;


typedef unsigned int  ELEMENT;     // type of array elements
typedef long          INDEX;       // type of array indices



/* CPU implementation of computing the sum of an array
 */
ELEMENT hAdd(INDEX n, const ELEMENT *A)
{
  ELEMENT sum = 0;
  
  for (INDEX i = 0; i < n; i++) {
    sum += A[i];
  }

  return sum;
}





/* This is the core of the GPU implementation of summing elements of the array inA.
   The array is divided into chunks, each of size numElementsPerBlock,
   which are each summed up by one thread block.
   The resulting sum is placed into the array outA.
   The position in outA is the block's index.
   As a result of running this kernel for all the blocks,
   the array outA is filled the all the partial sums,
   which then need to be summed up themselves.
   The array outA is smaller than inA by the factor numElementsPerBlock.

   This code assumes that numElementsPerBlock is twice
   the number of threads per block.
   And it assumes that numElementsPerBlock is a power of 2.
*/
__global__ void addKernel(INDEX inNum, const ELEMENT *inA,
			  INDEX outNum,      ELEMENT *outA)
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


 



/* The number of element per block must not exceed
   - the size of shared memory (into which the elements are copied)
   - double the number of threads available per block
   In addition, the element per block must be a power of 2,
   which is a requirement of the algorithm.
*/
INDEX calcNumElementsPerBlock()
{
  cudaDeviceProp p;   // structure with all the properties
  CU(cudaGetDeviceProperties(&p, 0));
  
  int allowed = min((int)(p.sharedMemPerBlock / sizeof(ELEMENT)),
		    2*min(p.maxThreadsDim[0],
			  p.maxThreadsPerBlock));
  
  
  // find closest power of 2 not-exceeding allowed

  INDEX result;
  for (result = 1;
       result < allowed;
       result <<= 1){
  }
  if (result > allowed){
    result >>= 1;
  }
  return result;
}




/*
  Sample invocation:
  ./ArrayAdd 1000
  where 1000 is the size of the array
*/
int main(int argc, char *argv[])
{
    
  // Read in the size of the array n.
  if (argc != 2) {
    cout << "ERROR:" << endl;
    cout << "Required argument: size of array" << endl;    
    cout << "Sample invocation:" << endl;
    cout << "    ./ArrayAdd 1000" << endl;        
    exit(1);
  }
        
  INDEX n = atol(argv[1]);



  
  //////////// CPU IMPLEMENTATION
  
  Timer cpuTimer(Timer::CPU);
  
  ELEMENT *hA = (ELEMENT *) malloc(n*sizeof(ELEMENT));  // array to add up

  if (hA == NULL) {
    printf("Not enough memory\n");
    exit(1);
  }

  // Fill it with random numbers
  srand(1);
  for (INDEX i = 0; i < n; i++) {
    hA[i] = rand();
  }

  cpuTimer.printElapsed("to allocate and initialize array");

  // Do the work
  ELEMENT hSum = hAdd(n, hA);

  cpuTimer.printElapsed("to add all elements");

  



  
  //////////// GPU IMPLEMENTATION

  Timer gpuTimer(Timer::GPU);
  
  // Transfer given array hA to an array dA on the device
  ELEMENT *dA;
  CU(cudaMalloc((void**)&dA, n*sizeof(ELEMENT)));
  CU(cudaMemcpy(dA, hA, n*sizeof(ELEMENT), cudaMemcpyHostToDevice));  

  CU(cudaDeviceSynchronize());
  gpuTimer.printElapsed("to allocate and transfer array");

  
  // Each block will add up a chunk of the array
  INDEX numElementsPerBlock =  calcNumElementsPerBlock();

  // The results from all the blocks are stored in another array (outA),
  // which then becomes subject to the same addition.
  // There is an alternation between the pair of arrays:
  // The algorithm operates on an inA (which is initially the original dA)
  // and stores the results into outA.
  // Then this outA becones inA for the next iteration, and
  // inA of the previous iteration becomes outA of the next.
  
  ELEMENT *inA = dA;
  INDEX    inNum = n;
  
  ELEMENT *outA;
  INDEX    outNum = (INDEX) ceil((float)inNum/numElementsPerBlock);
  CU(cudaMalloc((void**)&outA, outNum * sizeof(ELEMENT)));
    
  ELEMENT dSum;  // final result

  // Each iteration reduces the problem by a factor of numElementsPerBlock
  for(;;) {
    addKernel<<<outNum, numElementsPerBlock/2, numElementsPerBlock*sizeof(ELEMENT)>>>(inNum, inA, outNum, outA);

    // If this is the last iteration then extract the final result from outA
    if (outNum == 1) { 
      CU(cudaDeviceSynchronize());	
      CU(cudaMemcpy(&dSum, outA, sizeof(ELEMENT), cudaMemcpyDeviceToHost)); 
      break;
    }
    
    // Switch the roles of inA and outA for the next iteration
    inNum = outNum;
    outNum = (INDEX) ceil((float)inNum/numElementsPerBlock);
    
    ELEMENT *temp = inA;
    inA = outA;
    outA = temp;
  }
  
  CU(cudaDeviceSynchronize());
  gpuTimer.printElapsed("to add all elements");


  // Check whether cpu and GPU gave the same sum
  if (hSum != dSum) {
   cout << "ERROR: CPU sum = " << hSum << ", GPU sum = " << dSum << endl;
  }

  
#ifdef __CUDA__
  cout << "__CUDA__ is defined" << endl;
#endif
  
#ifdef __NVCC__
  cout << "__NVCC__ is defined" << endl;
#endif
}
