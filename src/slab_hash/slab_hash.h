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
#include "../memory_alloc/slab_list_alloc.h"
#include "coordinate.h"

/*
 * This is the main class that will be shallowly copied into the device to be
 * used at runtime. This class does not own the allocated memory on the gpu
 * (i.e., d_table_)
 */
template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
class SlabHashContext {
public:
    typedef Coordinate<KeyT, D> KeyTD;

public:
    SlabHashContext();
    static size_t SlabUnitSize();

    /** Initializer **/
    __host__ void Init(const uint32_t num_buckets,
                       int8_t* d_table,
                       const SlabAllocContext& allocator_ctx,
                       const MemoryAllocContext<KeyTD>& key_allocator_ctx,
                       const MemoryAllocContext<ValueT>& value_allocator_ctx);

    /** Core SIMT operations **/
    __device__ void InsertPair(bool& to_be_inserted,
                               const uint32_t& lane_id,
                               const KeyTD& myKey,
                               const ValueT& myValue,
                               const uint32_t bucket_id);
    __device__ void Search(bool& to_be_searched,
                           const uint32_t& lane_id,
                           const KeyTD& myKey,
                           ValueT& myValue,
                           bool& found,
                           const uint32_t bucket_id);
    __device__ void Delete(bool& to_be_deleted,
                           const uint32_t& lane_id,
                           const KeyTD& myKey,
                           const uint32_t bucket_id);

    /** Hash function **/
    __device__ __host__ __forceinline__ uint32_t
    ComputeBucket(const KeyTD& key) const;

    __device__ __forceinline__ int32_t laneFoundKeyInWarp(const KeyTD& src_key,
                                                          uint32_t lane_id,
                                                          uint32_t unit_data);

    __device__ __forceinline__ int32_t
    laneEmptyKeyInWarp(const uint32_t unit_data);

    __device__ __host__ __forceinline__ SlabAllocContext& getAllocatorContext();
    __device__ __host__ __forceinline__ ConcurrentSlab* getDeviceTablePointer();
    __device__ __forceinline__ uint32_t* getPointerFromSlab(
            const addr_t& slab_address, const uint32_t lane_id);
    __device__ __forceinline__ uint32_t* getPointerFromBucket(
            const uint32_t bucket_id, const uint32_t lane_id);

private:
    // this function should be operated in a warp-wide fashion
    // TODO: add required asserts to make sure this is true in tests/debugs
    __device__ __forceinline__ addr_t AllocateSlab(const uint32_t& lane_id);

    // a thread-wide function to free the slab that was just allocated
    __device__ __forceinline__ void FreeSlab(const addr_t slab_ptr);

private:
    uint32_t num_buckets_;
    HashFunc hash_fn_;

    ConcurrentSlab* d_table_;
    SlabAllocContext slab_list_allocator_ctx_;
    MemoryAllocContext<KeyTD> key_allocator_ctx_;
    MemoryAllocContext<ValueT> value_allocator_ctx_;
};

/*
 * This class owns the allocated memory for the hash table
 */
template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
class SlabHash {
public:
    typedef Coordinate<KeyT, D> KeyTD;

private:
    // fixed known parameters:
    static constexpr uint32_t BLOCKSIZE_ = 128;

    uint32_t num_buckets_;

    // a raw pointer to the initial allocated memory for all buckets
    int8_t* d_table_;
    size_t slab_unit_size_;

    SlabHashContext<KeyT, D, ValueT, HashFunc> gpu_context_;
    std::shared_ptr<MemoryAlloc<KeyTD>> key_allocator_;
    std::shared_ptr<MemoryAlloc<ValueT>> value_allocator_;
    std::shared_ptr<SlabAlloc> slab_list_allocator_;

    uint32_t device_idx_;

public:
    SlabHash(const uint32_t num_buckets,
             const std::shared_ptr<SlabAlloc>& slab_list_allocator,
             const std::shared_ptr<MemoryAlloc<KeyTD>>& key_allocator,
             const std::shared_ptr<MemoryAlloc<ValueT>>& value_allocator,
             uint32_t device_idx);

    ~SlabHash();

    // returns some debug information about the slab hash
    std::string to_string();
    double ComputeLoadFactor(int flag);

    void Insert(KeyTD* d_key, ValueT* d_value, uint32_t num_keys);
    void Search(KeyTD* d_query,
                ValueT* d_value,
                uint8_t* d_result,
                uint32_t num_queries);
    void Delete(KeyTD* d_key, uint32_t num_keys);
};
