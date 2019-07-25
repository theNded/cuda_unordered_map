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
