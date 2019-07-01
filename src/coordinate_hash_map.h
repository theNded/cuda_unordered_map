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

#include "slab_hash/slab_hash.h"

/* Lightweight wrapper to handle host input */
/* KeyT a elementary types: int, long, etc. */
template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
class CoordinateHashMap {
public:
    typedef Coordinate<KeyT, D> KeyTD;

    CoordinateHashMap(uint32_t max_keys,
                      uint32_t num_buckets,
                      const uint32_t device_idx = 0);
    ~CoordinateHashMap();

    void Insert(const std::vector<KeyTD>& keys,
                const std::vector<ValueT>& values,
                float& time);
    void Search(const std::vector<KeyTD>& query_keys,
                std::vector<ValueT>& query_results,
                float& time);
    void Delete(const std::vector<KeyTD>& keys, float& time);
    float ComputeLoadFactor(int flag = 0);

private:
    uint32_t max_keys_;
    uint32_t num_buckets_;
    uint32_t cuda_device_idx_;

    /** Handled by CUDA **/
    KeyTD* key_buffer_;
    ValueT* value_buffer_;
    KeyTD* query_key_buffer_;
    ValueT* query_result_buffer_;

    std::shared_ptr<MemoryAlloc<KeyTD>> key_allocator_;
    std::shared_ptr<MemoryAlloc<ValueT>> value_allocator_;
    std::shared_ptr<SlabListAlloc> slab_list_allocator_;
    std::shared_ptr<SlabHash<KeyT, D, ValueT, HashFunc>> slab_hash_;
};
