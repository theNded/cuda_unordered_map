/**
 * Created by wei on 18-3-29.
 */

#pragma once

#include <memory>
#include <vector>
#include "allocator.h"

/**
 * Memory allocation and free are expensive on GPU.
 * (And are easy to overflow, I need to check the the reason.)
 *
 * Basically, we maintain one memory heap per data type.
 */

#include <assert.h>
#include "../helper_cuda.h"
#include "config.h"

#define _CUDA_DEBUG_ENABLE_ASSERTION
template <typename T>
class MemoryAllocContext {
public:
    T *data_;           /* [N] */
    ptr_t *heap_;       /* [N] */
    int *heap_counter_; /* [1] */

public:
    int max_capacity_;

public:
    /**
     * The @value array's size is FIXED.
     * The @heap array stores the addresses of the values.
     * Only the unallocated part is maintained.
     * (ONLY care about the heap above the heap counter. Below is meaningless.)
     * ---------------------------------------------------------------------
     * heap  ---Malloc-->  heap  ---Malloc-->  heap  ---Free(0)-->  heap
     * N-1                 N-1                  N-1                  N-1   |
     *  .                   .                    .                    .    |
     *  .                   .                    .                    .    |
     *  .                   .                    .                    .    |
     *  3                   3                    3                    3    |
     *  2                   2                    2 <-                 2    |
     *  1                   1 <-                 1                    0 <- |
     *  0 <- heap_counter   0                    0                    0
     */
    __device__ ptr_t Allocate() {
        int index = atomicAdd(heap_counter_, 1);
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(index < max_capacity_);
#endif
        return heap_[index];
    }

    __device__ void Free(ptr_t ptr) {
        int index = atomicSub(heap_counter_, 1);
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(index >= 1);
#endif
        heap_[index - 1] = ptr;
    }

    __device__ T &extract(ptr_t ptr) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(addr < max_capacity_);
#endif
        return data_[ptr];
    }

    __device__ const T &extract(ptr_t ptr) const {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(addr < max_capacity_);
#endif
        return data_[ptr];
    }

    /* Returns the real ptr that can be accessed (instead of the internal ptr)
     */
    __device__ T *extract_ext_ptr(ptr_t ptr) { return data_ + ptr; }
};

template <typename T>
__global__ void ResetMemoryAllocKernel(MemoryAllocContext<T> ctx) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ctx.max_capacity_) {
        ctx.data_[i] = T(); /* This is not required. */
        ctx.heap_[i] = i;
    }
}

template <typename T, class Alloc>
class MemoryAlloc {
public:
    int max_capacity_;
    MemoryAllocContext<T> gpu_context_;
    std::shared_ptr<Alloc> allocator_;

public:
    MemoryAlloc(int max_capacity) {
        allocator_ = std::make_shared<Alloc>();
        max_capacity_ = max_capacity;
        gpu_context_.max_capacity_ = max_capacity;

        gpu_context_.heap_counter_ =
                allocator_->template allocate<int>(size_t(1));
        gpu_context_.heap_ =
                allocator_->template allocate<ptr_t>(size_t(max_capacity_));
        gpu_context_.data_ =
                allocator_->template allocate<T>(size_t(max_capacity_));

        const int blocks = (max_capacity_ + 128 - 1) / 128;
        const int threads = 128;

        ResetMemoryAllocKernel<<<blocks, threads>>>(gpu_context_);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaGetLastError());

        int heap_counter = 0;
        CHECK_CUDA(cudaMemcpy(gpu_context_.heap_counter_, &heap_counter,
                              sizeof(int), cudaMemcpyHostToDevice));
    }

    ~MemoryAlloc() {
        allocator_->template deallocate<int>(gpu_context_.heap_counter_);
        allocator_->template deallocate<ptr_t>(gpu_context_.heap_);
        allocator_->template deallocate<T>(gpu_context_.data_);
    }

    std::vector<int> DownloadHeap() {
        std::vector<int> ret;
        ret.resize(max_capacity_);
        CHECK_CUDA(cudaMemcpy(ret.data(), gpu_context_.heap_,
                              sizeof(int) * max_capacity_,
                              cudaMemcpyDeviceToHost));
        return ret;
    }

    std::vector<T> DownloadValue() {
        std::vector<T> ret;
        ret.resize(max_capacity_);
        CHECK_CUDA(cudaMemcpy(ret.data(), gpu_context_.data_,
                              sizeof(T) * max_capacity_,
                              cudaMemcpyDeviceToHost));
        return ret;
    }

    int heap_counter() {
        int heap_counter;
        CHECK_CUDA(cudaMemcpy(&heap_counter, gpu_context_.heap_counter_,
                              sizeof(int), cudaMemcpyDeviceToHost));
        return heap_counter;
    }
};
