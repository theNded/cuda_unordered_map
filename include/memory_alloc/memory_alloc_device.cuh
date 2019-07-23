//
// Created by dongw1 on 7/1/19.
//

#pragma once

#include "memory_alloc.h"

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
template <typename T>
__device__ addr_t MemoryAllocContext<T>::Allocate() {
    int index = atomicAdd(heap_counter_, 1);
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(index < max_capacity_);
#endif
    return heap_[index];
}

template <typename T>
__device__ void MemoryAllocContext<T>::Free(addr_t addr) {
    int index = atomicSub(heap_counter_, 1);
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(index >= 1);
#endif
    heap_[index - 1] = addr;
}

template <typename T>
__device__ addr_t &MemoryAllocContext<T>::addr_on_heap(size_t index) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(index < max_capacity_);
#endif
    return heap_[index];
}

template <typename T>
__device__ T &MemoryAllocContext<T>::value_at(addr_t addr) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(addr < max_capacity_);
#endif
    return data_[addr];
}

template <typename T>
__device__ const T &MemoryAllocContext<T>::value_at(addr_t addr) const {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(addr < max_capacity_);
#endif
    return data_[addr];
}