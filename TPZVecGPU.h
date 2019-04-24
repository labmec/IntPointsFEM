//adapted from https://www.quantstart.com/articles/dev_array_A_Useful_Array_Class_for_CUDA

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
	TPZVecGPU()
        : start_(0),
          end_(0)
    {}


    // constructor
	TPZVecGPU(size_t size)
    {
        allocate(size);
    }
    // destructor
    ~TPZVecGPU()
    {
        free();
    }

    // resize the vector
    void Resize(size_t size)
    {
        free();
        allocate(size);
    }

    // get the size of the array
    size_t GetSize() const
    {
        return end_ - start_;
    }

    // get data
    const T* GetData() const
    {
        return start_;
    }

    T* GetData()
    {
        return start_;
    }

    // set
    void Set(const T* src, size_t size)
    {
        size_t min = std::min(size, GetSize());
        cudaError_t result = cudaMemcpy(start_, src, min * sizeof(T), cudaMemcpyHostToDevice);
        if (result != cudaSuccess)
        {
            throw std::runtime_error("failed to copy to device memory");
        }
    }
    // get
    void Get(T* dest, size_t size)
    {
        size_t min = std::min(size, GetSize());
        cudaError_t result = cudaMemcpy(dest, start_, min * sizeof(T), cudaMemcpyDeviceToHost);
        if (result != cudaSuccess)
        {
            throw std::runtime_error("failed to copy to host memory");
        }
    }

    // initialize with zero
    void Zero()
    {
        cudaError_t result = cudaMemset(start_, 0, GetSize() * sizeof(T));
        if (result != cudaSuccess)
        {
            throw std::runtime_error("failed to fill with zero");
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
