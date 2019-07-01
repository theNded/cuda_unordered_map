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
#include "config.h"

#define CUDA_DEBUG_ENABLE_ASSERTION
template <typename T>
class MemoryAllocContext {
public:
    T *data_;           /* [N] */
    addr_t *heap_;      /* [N] */
    int *heap_counter_; /* [1] */

public:
    int max_capacity_;

public:
    __device__ addr_t Allocate();
    __device__ void Free(addr_t addr);

    __device__ inline T *data() { return data_; }
    __device__ inline addr_t *heap() { return heap_; }
    __device__ inline int *heap_counter() { return heap_counter_; }

    __device__ addr_t &addr_on_heap(size_t index);
    __device__ T &value_at(addr_t addr);
    __device__ const T &value_at(addr_t addr) const;
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