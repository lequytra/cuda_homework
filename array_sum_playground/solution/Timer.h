#ifndef TIMER_H
#define TIMER_H


#include "cuda.h"
#include "CU.h"
#include <chrono>
#include <iostream>
#include <iomanip> 

using namespace std;




class Timer
{

 public:

  enum Device {CPU, GPU};
  
 private:

#ifdef __NVCC__
  chrono::system_clock::time_point start;
#else
  chrono::steady_clock::time_point start;
#endif
  Device device;

 public:
  
  Timer(Device d)
    {
      start = chrono::high_resolution_clock::now();
      device = d;
    }
  
  void printElapsed(const char *msg)
    {
      auto end = chrono::high_resolution_clock::now();
      auto elapsed = chrono::duration_cast<chrono::milliseconds>(end - start);
      
      cout  << setw(10) << elapsed.count() << " ms on " << (device == CPU ? "CPU " : "GPU ") << msg << endl;
      
      start = end;
    }

};

#endif