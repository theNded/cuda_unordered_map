/*
 * Copyright 2019 Saman Ashkiani
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
 */

#pragma once

#include <cassert>
#include <memory>

#include "../memory_alloc/memory_alloc.h"
#include "../memory_alloc/slab_alloc.h"

struct Slab {
    // 15 x 2 + 1 + 1
    paired_internal_ptr_t data[15];
    slab_ptr_t next_slab_ptr;
    uint32_t padding;
};

/*
 * This is the main class that will be shallowly copied into the device to be
 * used at runtime. This class does not own the allocated memory on the gpu
 * (i.e., bucket_list_head_)
 */

template <typename Key>
struct hash {
    __device__ __host__ uint64_t operator()(const Key& key) const {
        uint64_t hash = UINT64_C(14695981039346656037);

        const int chunks = sizeof(Key) / sizeof(int);
        for (size_t i = 0; i < chunks; ++i) {
            hash ^= ((int32_t*)(&key))[i];
            hash *= UINT64_C(1099511628211);
        }
        return hash;
    }
};

template <typename KeyT, typename ValueT, typename HashFunc>
class SlabHashContext {
public:
    SlabHashContext();
    __host__ void Setup(Slab* bucket_list_head,
                        const uint32_t num_buckets,
                        const SlabAllocContext& allocator_ctx,
                        const MemoryAllocContext<KeyT>& key_allocator_ctx,
                        const MemoryAllocContext<ValueT>& value_allocator_ctx);

    /* Core SIMT operations */
    __device__ void InsertPair(bool& lane_active,
                               const uint32_t lane_id,
                               const uint32_t bucket_id,
                               const KeyT& key,
                               const ValueT& value);

    __device__ void Search(bool& lane_active,
                           const uint32_t lane_id,
                           const uint32_t bucket_id,
                           const KeyT& key,
                           ValueT& value,
                           uint8_t& found);

    __device__ void Delete(bool& lane_active,
                           const uint32_t lane_id,
                           const uint32_t bucket_id,
                           const KeyT& key);

    /* Hash function */
    __device__ __host__ __forceinline__ uint32_t
    ComputeBucket(const KeyT& key) const;

    __device__ __forceinline__ void WarpSyncKey(const KeyT& key,
                                                const uint32_t lane_id,
                                                KeyT& ret);
    __device__ __forceinline__ int32_t WarpFindKey(const KeyT& src_key,
                                                   const uint32_t lane_id,
                                                   const uint32_t unit_data);
    __device__ __forceinline__ int32_t WarpFindEmpty(const uint32_t unit_data);

    __device__ __host__ __forceinline__ SlabAllocContext& get_slab_alloc_ctx() {
        return slab_list_allocator_ctx_;
    }

    __device__ __forceinline__ uint32_t* get_unit_ptr_from_list_nodes(
            const slab_ptr_t slab_ptr, const uint32_t lane_id) {
        return slab_list_allocator_ctx_.get_unit_ptr_from_slab(slab_ptr,
                                                               lane_id);
    }
    __device__ __forceinline__ uint32_t* get_unit_ptr_from_list_head(
            const uint32_t bucket_id, const uint32_t lane_id) {
        return reinterpret_cast<uint32_t*>(bucket_list_head_) +
               bucket_id * BASE_UNIT_SIZE + lane_id;
    }

private:
    __device__ __forceinline__ slab_ptr_t AllocateSlab(const uint32_t lane_id);
    __device__ __forceinline__ void FreeSlab(const slab_ptr_t slab_ptr);

private:
    uint32_t num_buckets_;
    HashFunc hash_fn_;

    Slab* bucket_list_head_;
    SlabAllocContext slab_list_allocator_ctx_;
    MemoryAllocContext<KeyT> key_allocator_ctx_;
    MemoryAllocContext<ValueT> value_allocator_ctx_;
};

/*
 * This class owns the allocated memory for the hash table
 */
template <typename KeyT, typename ValueT, typename HashFunc>
class SlabHash {
private:
    static constexpr uint32_t BLOCKSIZE_ = 128;

    uint32_t num_buckets_;

    Slab* bucket_list_head_;

    SlabHashContext<KeyT, ValueT, HashFunc> gpu_context_;
    std::shared_ptr<MemoryAlloc<KeyT>> key_allocator_;
    std::shared_ptr<MemoryAlloc<ValueT>> value_allocator_;
    std::shared_ptr<SlabAlloc> slab_list_allocator_;

    uint32_t device_idx_;

public:
    SlabHash(const uint32_t num_buckets,
             const std::shared_ptr<SlabAlloc>& slab_list_allocator,
             const std::shared_ptr<MemoryAlloc<KeyT>>& key_allocator,
             const std::shared_ptr<MemoryAlloc<ValueT>>& value_allocator,
             uint32_t device_idx);

    ~SlabHash();

    double ComputeLoadFactor(int flag);

    void Insert(KeyT* keys, ValueT* values, uint32_t num_keys);
    void Search(KeyT* keys,
                ValueT* values,
                uint8_t* founds,
                uint32_t num_queries);
    void Delete(KeyT* keys, uint32_t num_keys);
};
