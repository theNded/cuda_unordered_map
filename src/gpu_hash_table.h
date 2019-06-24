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

#include "memory_heap/memory_heap_host.cuh"
#include "slab_hash/instantiate.cuh"

/* Lightweight wrapper to handle host input */
/* KeyT a elementary types: int, long, etc. */
template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
class GpuHashTable {
public:
    typedef Coordinate<KeyT, D> KeyTD;

    GpuHashTable(uint32_t max_keys,
                 uint32_t num_buckets,
                 const uint32_t device_idx);
    ~GpuHashTable();

    float Insert(KeyTD* h_key, ValueT* h_value, uint32_t num_keys);
    float Search(KeyTD* h_query, ValueT* h_result, uint32_t num_queries);
    float Delete(KeyTD* h_key, uint32_t num_keys);
    float ComputeLoadFactor(int flag = 0);

private:
    uint32_t max_keys_;
    uint32_t num_buckets_;
    int64_t seed_;
    bool req_values_;
    bool identity_hash_;
    uint32_t cuda_device_idx_;

    KeyTD* d_key_;
    ValueT* d_value_;
    KeyTD* d_query_;
    ValueT* d_result_;

    std::shared_ptr<MemoryHeap<KeyTD>> key_allocator_;
    std::shared_ptr<MemoryHeap<ValueT>> value_allocator_;
    std::shared_ptr<SlabListAllocator> slab_list_allocator_;
    std::shared_ptr<GpuSlabHash<KeyT, D, ValueT, HashFunc>> slab_hash_;
};
