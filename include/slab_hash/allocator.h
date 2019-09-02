#include "../helper_cuda.h"
#include "config.h"

class Allocator {
public:
    Allocator(int device_id = 0) : device_id_(device_id) {}
    template <typename T>
    virtual T* allocate(size_t size) = 0;

    template <typename T>
    virtual void free(T* ptr) = 0;

protected:
    int device_id_;
};

class CudaAllocator : public Allocator {
public:
    CudaAllocator(int device_id = 0) : Allocator(device_id) {}
    template <typename T>
    T* allocate(size_t size) override {
        T* ptr;
        CHECK_CUDA(cudaMalloc(&ptr, sizeof(T) * size));
        return ptr;
    }

    template <typename T>
    void free(T* ptr) override {
        CHECK_CUDA(cudaFree(ptr));
    }
};

/**
class PyTorchAllocator : public Allocator {
public:
    PyTorchAllocator(int device_id = 0) : Allocator(device_id) {}

    template <typename T>
    T* allocate(size_t size) override {
        CHECK_CUDA(cudaGetDevice(&device_id_));
        auto options = torch::TensorOptions()
                               .dtype(torch::kInt8)
                               .device(torch::kCUDA, device_id_)
                               .requires_grad(false);
        tensor_ = torch::zeros(sizeof(T) * size, options);
        return tensor_.data<T>()
    }

    template <typename T>
    void free(T* ptr) override {
        // let PyTorch handle this
    }

protected:
    torch::Tensor tensor_;
};
**/
