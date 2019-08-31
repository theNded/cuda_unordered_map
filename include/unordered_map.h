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

#include <thrust/device_vector.h>
#include "slab_hash/slab_hash.h"

/* Lightweight wrapper to handle host input */
/* KeyT supports elementary types: int, long, etc. */
/* ValueT supports arbitrary types in theory. */
template <typename KeyT, typename ValueT, typename HashFunc = hash<KeyT>>
class CoordinateHashMap {
public:
    CoordinateHashMap(uint32_t max_keys,
                      /* Preset hash table params to estimate bucket num */
                      uint32_t keys_per_bucket = 15,
                      float expected_occupancy_per_bucket = 0.6,
                      /* CUDA device */
                      const uint32_t device_idx = 0);
    ~CoordinateHashMap();

    /* We assert all memory buffers are allocated prior to the function call
         @keys_device stores keys in KeyT[num_keys x D],
         @[query]_values_device stores keys in ValueT[num_keys] */
    /* query_values[i] is undefined (basically ValueT(0)) if mask[i] == 0 */

    float Insert(thrust::device_vector<KeyT>& keys,
                 thrust::device_vector<ValueT>& values);
    float Insert(const std::vector<KeyT>& keys,
                 const std::vector<ValueT>& values);
    float Insert(KeyT* keys_device, ValueT* values_device, int num_keys);

float Search(thrust::device_vector<KeyT>& query_keys,
                 thrust::device_vector<ValueT>& query_values,
                 thrust::device_vector<uint8_t>& mask);
    float Search(const std::vector<KeyT>& query_keys,
                 std::vector<ValueT>& query_values,
                 std::vector<uint8_t>& mask);
    float Search(KeyT* query_keys_device,
                 ValueT* query_values_device,
                 uint8_t* mask,
                 int num_keys);

    float Delete(thrust::device_vector<KeyT>& keys);
    float Delete(const std::vector<KeyT>& keys);
    float Delete(KeyT* keys, int num_keys);

    float ComputeLoadFactor(int flag = 0);

private:
    uint32_t max_keys_;
    uint32_t num_buckets_;
    uint32_t cuda_device_idx_;

    /* Timer */
    cudaEvent_t start_;
    cudaEvent_t stop_;

    /* Handled by CUDA */
    KeyT* key_buffer_;
    ValueT* value_buffer_;
    KeyT* query_key_buffer_;
    ValueT* query_value_buffer_;
    uint8_t* query_result_buffer_;

    /* Context manager */
    std::shared_ptr<MemoryAlloc<thrust::pair<KeyT, ValueT>>> pair_allocator_;
    std::shared_ptr<SlabAlloc> slab_list_allocator_;

    std::shared_ptr<SlabHash<KeyT, ValueT, HashFunc>> slab_hash_;
};

template <typename KeyT, typename ValueT, typename HashFunc>
CoordinateHashMap<KeyT, ValueT, HashFunc>::CoordinateHashMap(
        uint32_t max_keys,
        uint32_t keys_per_bucket,
        float expected_occupancy_per_bucket,
        const uint32_t device_idx)
    : max_keys_(max_keys),
      cuda_device_idx_(device_idx),
      slab_hash_(nullptr),
      slab_list_allocator_(nullptr) {
    /* Set bucket size */
    uint32_t expected_keys_per_bucket =
            expected_occupancy_per_bucket * keys_per_bucket;
    num_buckets_ = (max_keys + expected_keys_per_bucket - 1) /
                   expected_keys_per_bucket;

    /* Set device */
    int32_t cuda_device_count_ = 0;
    CHECK_CUDA(cudaGetDeviceCount(&cuda_device_count_));
    assert(cuda_device_idx_ < cuda_device_count_);
    CHECK_CUDA(cudaSetDevice(cuda_device_idx_));

    // allocating key, value arrays:
    CHECK_CUDA(cudaMalloc(&key_buffer_, sizeof(KeyT) * max_keys_));
    CHECK_CUDA(cudaMalloc(&value_buffer_, sizeof(ValueT) * max_keys_));
    CHECK_CUDA(cudaMalloc(&query_key_buffer_, sizeof(KeyT) * max_keys_));
    CHECK_CUDA(cudaMalloc(&query_value_buffer_, sizeof(ValueT) * max_keys_));
    CHECK_CUDA(cudaMalloc(&query_result_buffer_, sizeof(uint8_t) * max_keys_));

    CHECK_CUDA(cudaEventCreate(&start_));
    CHECK_CUDA(cudaEventCreate(&stop_));

    // allocate an initialize the allocator:
    pair_allocator_ = std::make_shared<MemoryAlloc<thrust::pair<KeyT, ValueT>>>(
            max_keys_);
    slab_list_allocator_ = std::make_shared<SlabAlloc>();
    slab_hash_ = std::make_shared<SlabHash<KeyT, ValueT, HashFunc>>(
            num_buckets_, slab_list_allocator_, pair_allocator_,
            cuda_device_idx_);
}

template <typename KeyT, typename ValueT, typename HashFunc>
CoordinateHashMap<KeyT, ValueT, HashFunc>::~CoordinateHashMap() {
    CHECK_CUDA(cudaSetDevice(cuda_device_idx_));

    CHECK_CUDA(cudaFree(key_buffer_));
    CHECK_CUDA(cudaFree(value_buffer_));

    CHECK_CUDA(cudaFree(query_key_buffer_));
    CHECK_CUDA(cudaFree(query_value_buffer_));

    CHECK_CUDA(cudaEventDestroy(start_));
    CHECK_CUDA(cudaEventDestroy(stop_));
}

template <typename KeyT, typename ValueT, typename HashFunc>
float CoordinateHashMap<KeyT, ValueT, HashFunc>::Insert(
        const std::vector<KeyT>& keys, const std::vector<ValueT>& values) {
    float time;
    assert(values.size() == keys.size());

    CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
    CHECK_CUDA(cudaMemcpy(key_buffer_, keys.data(), sizeof(KeyT) * keys.size(),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(value_buffer_, values.data(),
                          sizeof(ValueT) * values.size(),
                          cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaEventRecord(start_, 0));

    slab_hash_->Insert(key_buffer_, value_buffer_, keys.size());

    CHECK_CUDA(cudaEventRecord(stop_, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_));
    CHECK_CUDA(cudaEventElapsedTime(&time, start_, stop_));
    return time;
}

template <typename KeyT, typename ValueT, typename HashFunc>
float CoordinateHashMap<KeyT, ValueT, HashFunc>::Insert(
        thrust::device_vector<KeyT>& keys,
        thrust::device_vector<ValueT>& values) {
    float time;
    assert(values.size() == keys.size());

    CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
    CHECK_CUDA(cudaEventRecord(start_, 0));

    slab_hash_->Insert(thrust::raw_pointer_cast(keys.data()),
                       thrust::raw_pointer_cast(values.data()), keys.size());

    CHECK_CUDA(cudaEventRecord(stop_, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_));
    CHECK_CUDA(cudaEventElapsedTime(&time, start_, stop_));
    return time;
}

template <typename KeyT, typename ValueT, typename HashFunc>
float CoordinateHashMap<KeyT, ValueT, HashFunc>::Insert(KeyT* keys,
                                                        ValueT* values,
                                                        int num_keys) {
    float time;
    CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
    CHECK_CUDA(cudaEventRecord(start_, 0));
    slab_hash_->Insert(keys, values, num_keys);
    CHECK_CUDA(cudaEventRecord(stop_, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_));
    CHECK_CUDA(cudaEventElapsedTime(&time, start_, stop_));
    return time;
}

template <typename KeyT, typename ValueT, typename HashFunc>
float CoordinateHashMap<KeyT, ValueT, HashFunc>::Search(
        const std::vector<KeyT>& query_keys,
        std::vector<ValueT>& query_values,
        std::vector<uint8_t>& query_found) {
    float time;
    assert(query_found.size() >= query_keys.size());
    assert(query_values.size() >= query_keys.size());

    CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
    CHECK_CUDA(cudaMemcpy(query_key_buffer_, query_keys.data(),
                          sizeof(KeyT) * query_keys.size(),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(query_value_buffer_, 0xFF,
                          sizeof(ValueT) * query_keys.size()));
    CHECK_CUDA(cudaMemset(query_result_buffer_, 0,
                          sizeof(uint8_t) * query_keys.size()));

    CHECK_CUDA(cudaEventRecord(start_, 0));

    slab_hash_->Search(query_key_buffer_, query_value_buffer_,
                       query_result_buffer_, query_keys.size());

    CHECK_CUDA(cudaEventRecord(stop_, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_));
    CHECK_CUDA(cudaEventElapsedTime(&time, start_, stop_));

    CHECK_CUDA(cudaMemcpy(query_values.data(), query_value_buffer_,
                          sizeof(ValueT) * query_keys.size(),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(query_found.data(), query_result_buffer_,
                          sizeof(uint8_t) * query_keys.size(),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDeviceSynchronize());
    return time;
}

template <typename KeyT, typename ValueT, typename HashFunc>
float CoordinateHashMap<KeyT, ValueT, HashFunc>::Search(
        thrust::device_vector<KeyT>& query_keys,
        thrust::device_vector<ValueT>& query_values,
        thrust::device_vector<uint8_t>& mask) {
    float time;
    CHECK_CUDA(cudaSetDevice(cuda_device_idx_));

    thrust::fill(mask.begin(), mask.end(), 0);
    CHECK_CUDA(cudaEventRecord(start_, 0));

    slab_hash_->Search(thrust::raw_pointer_cast(query_keys.data()),
                       thrust::raw_pointer_cast(query_values.data()),
                       thrust::raw_pointer_cast(mask.data()),
                       query_keys.size());

    CHECK_CUDA(cudaEventRecord(stop_, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_));
    CHECK_CUDA(cudaEventElapsedTime(&time, start_, stop_));
    return time;
}

template <typename KeyT, typename ValueT, typename HashFunc>
float CoordinateHashMap<KeyT, ValueT, HashFunc>::Search(KeyT* query_keys,
                                                        ValueT* query_values,
                                                        uint8_t* mask,
                                                        int num_keys) {
    float time;
    CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
    CHECK_CUDA(cudaMemset(mask, 0, sizeof(uint8_t) * num_keys));
    CHECK_CUDA(cudaEventRecord(start_, 0));

    slab_hash_->Search(query_keys, query_values, mask, num_keys);

    CHECK_CUDA(cudaEventRecord(stop_, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_));
    CHECK_CUDA(cudaEventElapsedTime(&time, start_, stop_));
    return time;
}

template <typename KeyT, typename ValueT, typename HashFunc>
float CoordinateHashMap<KeyT, ValueT, HashFunc>::Delete(
        const std::vector<KeyT>& keys) {
    float time;
    CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
    CHECK_CUDA(cudaMemcpy(key_buffer_, keys.data(), sizeof(KeyT) * keys.size(),
                          cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaEventRecord(start_, 0));

    slab_hash_->Delete(key_buffer_, keys.size());

    CHECK_CUDA(cudaEventRecord(stop_, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_));
    CHECK_CUDA(cudaEventElapsedTime(&time, start_, stop_));
    return time;
}

template <typename KeyT, typename ValueT, typename HashFunc>
float CoordinateHashMap<KeyT, ValueT, HashFunc>::Delete(
        thrust::device_vector<KeyT>& keys) {
    float time;
    CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
    CHECK_CUDA(cudaEventRecord(start_, 0));

    slab_hash_->Delete(thrust::raw_pointer_cast(keys.data()), keys.size());
    CHECK_CUDA(cudaEventRecord(stop_, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_));
    CHECK_CUDA(cudaEventElapsedTime(&time, start_, stop_));

    return time;
}

template <typename KeyT, typename ValueT, typename HashFunc>
float CoordinateHashMap<KeyT, ValueT, HashFunc>::Delete(KeyT* keys,
                                                        int num_keys) {
    float time;
    CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
    CHECK_CUDA(cudaEventRecord(start_, 0));

    slab_hash_->Delete(keys, num_keys);
    CHECK_CUDA(cudaEventRecord(stop_, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_));
    CHECK_CUDA(cudaEventElapsedTime(&time, start_, stop_));

    return time;
}

template <typename KeyT, typename ValueT, typename HashFunc>
float CoordinateHashMap<KeyT, ValueT, HashFunc>::ComputeLoadFactor(
        int flag /* = 0 */) {
    return slab_hash_->ComputeLoadFactor(flag);
}

