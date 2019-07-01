/**
 * Created by wei on 18-3-29.
 */

#pragma once

#include <memory>
#include <vector>

/**
 * Memory allocation and free are expensive on GPU.
 * (And are easy to overflow, I need to check the the reason.)
 *
 * Basically, we maintain one memory heap per data type.
 */

#include <assert.h>

#define CUDA_DEBUG_ENABLE_ASSERTION
template <typename T>
class MemoryAllocContext {
public:
    T *data_;           /* [N] */
    int *heap_;         /* [N] */
    int *heap_counter_; /* [1] */

public:
    int max_capacity_;

public:
    __device__ inline T *data() { return data_; }
    __device__ inline int *heap() { return heap_; }
    __device__ inline int *heap_counter() { return heap_counter_; }

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
    __device__ int Malloc();
    __device__ void Free(size_t addr);
    __device__ int &internal_addr_at(size_t index);
    __device__ T &value_at(size_t addr);
    __device__ const T &value_at(size_t addr) const;
};

template <typename T>
class MemoryAlloc {
public:
    int heap_counter();

public:
    int max_capacity_;
    MemoryAllocContext<T> gpu_context_;

public:
    MemoryAlloc(int max_capacity);
    ~MemoryAlloc();

    /* Hopefully this is only used for debugging. */
    std::vector<int> DownloadHeap();
    std::vector<T> DownloadValue();
};