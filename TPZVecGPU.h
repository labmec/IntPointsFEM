#ifndef _TPZVECGPU_H_
#define _TPZVECGPU_H_

#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s line:%d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template <class T>
class TPZVecGPU
{
// public functions
public:
    explicit TPZVecGPU()
        : start_(0),
          end_(0)
    {}

    // constructor
    explicit TPZVecGPU(size_t size)
    {
        allocate(size);
    }
    // destructor
    ~TPZVecGPU()
    {
        free();
    }

    // resize the vector
    void resize(size_t size)
    {
        free();
        allocate(size);
    }

    // get the size of the array
    size_t getSize() const
    {
        return end_ - start_;
    }

    // get data
    const T* getData() const
    {
        return start_;
    }

    T* getData()
    {
        return start_;
    }

    void Zero() { 
        gpuErrchk(cudaMemset(start_, 0, getSize() * sizeof(T)));
    }

    // set
    void set(const T* src) {
        gpuErrchk(cudaMemcpy(start_, src, getSize() * sizeof(T), cudaMemcpyHostToDevice));
    }

    // get
   void get(T* dest, size_t size) {
        size_t min = std::min(size, getSize());
        gpuErrchk(cudaMemcpy(dest, start_, min * sizeof(T), cudaMemcpyDeviceToHost)); 
    }


// private functions
private:
    // allocate memory on the device
    void allocate(size_t size) {
        gpuErrchk(cudaMalloc((void**)&start_, size * sizeof(T)));
        end_ = start_ + size;
    }

    // free memory on the device
    void free() {
        if (start_ != 0) {
            cudaFree(start_);
            start_ = end_ = 0;
        }
    }

    T* start_;
    T* end_;
};

#endif