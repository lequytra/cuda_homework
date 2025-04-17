#ifndef CU_H
#define CU_H

#include <stdlib.h> 
#include <stdio.h>

#define CU(call) {                                                                                     \
cudaError_t err = cudaGetLastError();                                                                  \
if( cudaSuccess != err) {                                                                              \
  fprintf(stderr, "%s, line %d: BEFORE REACHED %s.\n", __FILE__, __LINE__, cudaGetErrorString(err) );  \
  exit(-1);                                                                                            \
 }                                                                                                     \
err = call;                                                                                            \
if( cudaSuccess != err) {                                                                              \
  fprintf(stderr, "%s, line %d: %s.\n", __FILE__, __LINE__, cudaGetErrorString(err) );                 \
  exit(-1);                                                                                            \
 }                                                                                                     \
}

#endif