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
#include "gpu_hash_table.h"

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
GpuHashTable<KeyT, D, ValueT, HashFunc>::GpuHashTable(uint32_t max_keys,
                                                      uint32_t num_buckets,
                                                      const uint32_t device_idx)
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
    CHECK_CUDA(cudaMalloc((void**)&d_key_, sizeof(KeyTD) * max_keys_));
    CHECK_CUDA(cudaMalloc((void**)&d_value_, sizeof(ValueT) * max_keys_));
    CHECK_CUDA(cudaMalloc((void**)&d_query_, sizeof(KeyTD) * max_keys_));
    CHECK_CUDA(cudaMalloc((void**)&d_result_, sizeof(ValueT) * max_keys_));

    // allocate an initialize the allocator:
    slab_list_allocator_ = std::make_shared<SlabListAllocator>();
    key_allocator_ = std::make_shared<MemoryHeap<KeyTD>>(max_keys_);
    value_allocator_ = std::make_shared<MemoryHeap<ValueT>>(max_keys_);

    slab_hash_ = std::make_shared<GpuSlabHash<KeyT, D, ValueT, HashFunc>>(
            num_buckets_, slab_list_allocator_, key_allocator_,
            value_allocator_, cuda_device_idx_);
}

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
GpuHashTable<KeyT, D, ValueT, HashFunc>::~GpuHashTable() {
    CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
    CHECK_CUDA(cudaFree(d_key_));

    CHECK_CUDA(cudaFree(d_value_));

    CHECK_CUDA(cudaFree(d_query_));
    CHECK_CUDA(cudaFree(d_result_));
}

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
float GpuHashTable<KeyT, D, ValueT, HashFunc>::Insert(KeyTD* h_key,
                                                      ValueT* h_value,
                                                      uint32_t num_keys) {
    // moving key-values to the device:
    CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
    CHECK_CUDA(cudaMemcpy(d_key_, h_key, sizeof(KeyTD) * num_keys,
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_value_, h_value, sizeof(ValueT) * num_keys,
                          cudaMemcpyHostToDevice));

    float temp_time = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    slab_hash_->Insert(d_key_, d_value_, num_keys);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&temp_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return temp_time;
}

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
float GpuHashTable<KeyT, D, ValueT, HashFunc>::Search(KeyTD* h_query,
                                                      ValueT* h_result,
                                                      uint32_t num_queries) {
    CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
    CHECK_CUDA(cudaMemcpy(d_query_, h_query, sizeof(KeyTD) * num_queries,
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_result_, 0xFF, sizeof(ValueT) * num_queries));

    float temp_time = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    slab_hash_->Search(d_query_, d_result_, num_queries);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&temp_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CHECK_CUDA(cudaMemcpy(h_result, d_result_, sizeof(ValueT) * num_queries,
                          cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    return temp_time;
}

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
float GpuHashTable<KeyT, D, ValueT, HashFunc>::Delete(KeyTD* h_key,
                                                      uint32_t num_keys) {
    CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
    CHECK_CUDA(cudaMemcpy(d_key_, h_key, sizeof(KeyTD) * num_keys,
                          cudaMemcpyHostToDevice));

    float temp_time = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    slab_hash_->Delete(d_key_, num_keys);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&temp_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return temp_time;
}

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
float GpuHashTable<KeyT, D, ValueT, HashFunc>::ComputeLoadFactor(
        int flag /* = 0 */) {
    return slab_hash_->computeLoadFactor(flag);
}