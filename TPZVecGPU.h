#ifndef _TPZVECGPU_H_
#define _TPZVECGPU_H_

#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>

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

    void fill(T value)
    { 
        cudaError_t result = cudaMemset(start_, value, getSize() * sizeof(T));
        if (result != cudaSuccess)
        {
            throw std::runtime_error("failed to copy to device memory");
        }
    }



    // set
    void set(const T* src)
    {
        cudaError_t result = cudaMemcpy(start_, src, getSize() * sizeof(T), cudaMemcpyHostToDevice);
        if (result != cudaSuccess)
        {
            throw std::runtime_error("failed to copy to device memory");
        }
    }
    // get
    void get(T* dest)
    {
        cudaError_t result = cudaMemcpy(dest, start_, getSize() * sizeof(T), cudaMemcpyDeviceToHost);
        if (result != cudaSuccess)
        {
            throw std::runtime_error("failed to copy to host memory");
        }
    }


// private functions
private:
    // allocate memory on the device
    void allocate(size_t size)
    {
        cudaError_t result = cudaMalloc((void**)&start_, size * sizeof(T));
        if (result != cudaSuccess)
        {
            start_ = end_ = 0;
            throw std::runtime_error("failed to allocate device memory");
        }
        end_ = start_ + size;
    }

    // free memory on the device
    void free()
    {
        if (start_ != 0)
        {
            cudaFree(start_);
            start_ = end_ = 0;
        }
    }

    T* start_;
    T* end_;
};

#endif