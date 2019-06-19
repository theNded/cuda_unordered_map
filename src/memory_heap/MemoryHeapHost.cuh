//
// Created by wei on 18-11-9.
//

#pragma once

#include "../helper/helper_cuda.h"
#include "MemoryHeap.h"

#include <cassert>

/**
 * Client end
 */
template <typename T>
MemoryHeap<T>::MemoryHeap(int max_capacity) {
    max_capacity_ = max_capacity;
    gpu_context_.max_capacity_ = max_capacity;
    CHECK_CUDA(cudaMalloc(&(gpu_context_.heap_counter_), sizeof(int)));
    CHECK_CUDA(cudaMalloc(&(gpu_context_.heap_), sizeof(int) * max_capacity_));
    CHECK_CUDA(cudaMalloc(&(gpu_context_.data_), sizeof(T) * max_capacity_));
}

template <typename T>
MemoryHeap<T>::~MemoryHeap() {
    CHECK_CUDA(cudaFree(gpu_context_.heap_counter_));
    CHECK_CUDA(cudaFree(gpu_context_.heap_));
    CHECK_CUDA(cudaFree(gpu_context_.data_));
}

template <typename T>
std::vector<int> MemoryHeap<T>::DownloadHeap() {
    std::vector<int> ret;
    ret.resize(max_capacity_);
    CHECK_CUDA(cudaMemcpy(ret.data(), gpu_context_.heap_,
                         sizeof(int) * max_capacity_, cudaMemcpyDeviceToHost));
    return ret;
}

template <typename T>
std::vector<T> MemoryHeap<T>::DownloadValue() {
    std::vector<T> ret;
    ret.resize(max_capacity_);
    CHECK_CUDA(cudaMemcpy(ret.data(), gpu_context_.data_,
                         sizeof(T) * max_capacity_, cudaMemcpyDeviceToHost));
    return ret;
}

template <typename T>
int MemoryHeap<T>::HeapCounter() {
    int heap_counter;
    CHECK_CUDA(cudaMemcpy(&heap_counter, gpu_context_.heap_counter_, sizeof(int),
                         cudaMemcpyDeviceToHost));
    return heap_counter;
}
