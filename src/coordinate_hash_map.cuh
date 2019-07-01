/*
 * Copyright 2019 Saman Ashkiani
 * Modified by Wei Dong (2019)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#pragma once

#include "coordinate_hash_map.h"
#include "memory_alloc/memory_alloc_host.cuh"
#include "slab_hash/slab_hash_host.cuh"

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
CoordinateHashMap<KeyT, D, ValueT, HashFunc>::CoordinateHashMap(
        uint32_t max_keys, uint32_t num_buckets, const uint32_t device_idx)
    : max_keys_(max_keys),
      num_buckets_(num_buckets),
      cuda_device_idx_(device_idx),
      slab_hash_(nullptr),
      slab_list_allocator_(nullptr) {
    /* Set device */
    int32_t cuda_device_count_ = 0;
    CHECK_CUDA(cudaGetDeviceCount(&cuda_device_count_));
    assert(cuda_device_idx_ < cuda_device_count_);
    CHECK_CUDA(cudaSetDevice(cuda_device_idx_));

    // allocating key, value arrays:
    CHECK_CUDA(cudaMalloc(&key_buffer_, sizeof(KeyTD) * max_keys_));
    CHECK_CUDA(cudaMalloc(&value_buffer_, sizeof(ValueT) * max_keys_));
    CHECK_CUDA(cudaMalloc(&query_key_buffer_, sizeof(KeyTD) * max_keys_));
    CHECK_CUDA(cudaMalloc(&query_result_buffer_, sizeof(ValueT) * max_keys_));

    // allocate an initialize the allocator:
    key_allocator_ = std::make_shared<MemoryAlloc<KeyTD>>(max_keys_);
    value_allocator_ = std::make_shared<MemoryAlloc<ValueT>>(max_keys_);
    slab_list_allocator_ = std::make_shared<SlabAlloc>();
    slab_hash_ = std::make_shared<SlabHash<KeyT, D, ValueT, HashFunc>>(
            num_buckets_, slab_list_allocator_, key_allocator_,
            value_allocator_, cuda_device_idx_);
}

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
CoordinateHashMap<KeyT, D, ValueT, HashFunc>::~CoordinateHashMap() {
    CHECK_CUDA(cudaSetDevice(cuda_device_idx_));

    CHECK_CUDA(cudaFree(key_buffer_));
    CHECK_CUDA(cudaFree(value_buffer_));

    CHECK_CUDA(cudaFree(query_key_buffer_));
    CHECK_CUDA(cudaFree(query_result_buffer_));
}

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
void CoordinateHashMap<KeyT, D, ValueT, HashFunc>::Insert(
        const std::vector<KeyTD>& keys,
        const std::vector<ValueT>& values,
        float& time) {
    assert(values.size() == keys.size());

    CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
    CHECK_CUDA(cudaMemcpy(key_buffer_, keys.data(), sizeof(KeyTD) * keys.size(),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(value_buffer_, values.data(),
                          sizeof(ValueT) * values.size(),
                          cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    slab_hash_->Insert(key_buffer_, value_buffer_, keys.size());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
void CoordinateHashMap<KeyT, D, ValueT, HashFunc>::Search(
        const std::vector<KeyTD>& query_keys,
        std::vector<ValueT>& query_results,
        float& time) {
    assert(query_results.size() >= query_keys.size());

    CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
    CHECK_CUDA(cudaMemcpy(query_key_buffer_, query_keys.data(),
                          sizeof(KeyTD) * query_keys.size(),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(query_result_buffer_, 0xFF,
                          sizeof(ValueT) * query_results.size()));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    slab_hash_->Search(query_key_buffer_, query_result_buffer_,
                       query_keys.size());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CHECK_CUDA(cudaMemcpy(query_results.data(), query_result_buffer_,
                          sizeof(ValueT) * query_results.size(),
                          cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
}

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
void CoordinateHashMap<KeyT, D, ValueT, HashFunc>::Delete(
        const std::vector<KeyTD>& keys, float& time) {
    CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
    CHECK_CUDA(cudaMemcpy(key_buffer_, keys.data(), sizeof(KeyTD) * keys.size(),
                          cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    slab_hash_->Delete(key_buffer_, keys.size());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
float CoordinateHashMap<KeyT, D, ValueT, HashFunc>::ComputeLoadFactor(
        int flag /* = 0 */) {
    return slab_hash_->computeLoadFactor(flag);
}