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

#include "slab_hash.h"
#include "../memory_alloc/slab_list_alloc.cuh"
#include "../memory_alloc/memory_alloc_ctx.cuh"

// fixed known parameters:
template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
SlabHashContext<KeyT, D, ValueT, HashFunc>::SlabHashContext()
    : num_buckets_(0), d_table_(nullptr) {
    // a single slab on a ConcurrentMap should be 128 bytes
    assert(sizeof(ConcurrentSlab) == (WARP_WIDTH * sizeof(uint32_t)));
}

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
size_t SlabHashContext<KeyT, D, ValueT, HashFunc>::SlabUnitSize() {
    return sizeof(ConcurrentSlab);
}

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
__host__ void SlabHashContext<KeyT, D, ValueT, HashFunc>::Init(
        const uint32_t num_buckets,
        int8_t* d_table,
        SlabListAllocContext* allocator_ctx,
        MemoryAllocContext<KeyTD> key_allocator_ctx,
        MemoryAllocContext<ValueT> value_allocator_ctx) {
    num_buckets_ = num_buckets;
    d_table_ = reinterpret_cast<ConcurrentSlab*>(d_table);
    slab_list_allocator_ctx_ = *allocator_ctx;
    key_allocator_ctx_ = key_allocator_ctx;
    value_allocator_ctx_ = value_allocator_ctx;
}

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
__device__ __host__ __forceinline__ uint32_t
SlabHashContext<KeyT, D, ValueT, HashFunc>::ComputeBucket(
        const KeyTD& key) const {
    return hash_fn_(key) % num_buckets_;
}

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
__device__ int32_t
SlabHashContext<KeyT, D, ValueT, HashFunc>::laneFoundKeyInWarp(
        const KeyTD& src_key, uint32_t lane_id, uint32_t unit_data) {
    bool is_lane_found =
            /* select key lanes */
            ((1 << lane_id) & REGULAR_NODE_KEY_MASK)
            /* validate key addrs */
            && (unit_data != EMPTY_KEY)
            /* find keys in memory heap */
            && key_allocator_ctx_.value_at(unit_data) == src_key;

    return __ffs(__ballot_sync(REGULAR_NODE_KEY_MASK, is_lane_found)) - 1;
}

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
__device__ __forceinline__ int32_t
SlabHashContext<KeyT, D, ValueT, HashFunc>::laneEmptyKeyInWarp(
        const uint32_t unit_data) {
    uint32_t isEmpty = __ballot_sync(0xFFFFFFFF, (unit_data == EMPTY_KEY));
    return __ffs(isEmpty & REGULAR_NODE_KEY_MASK) - 1;
}


template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
__device__ __host__ __forceinline__ SlabListAllocContext&
SlabHashContext<KeyT, D, ValueT, HashFunc>::getAllocatorContext() {
    return slab_list_allocator_ctx_;
}

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
__device__ __host__ __forceinline__ ConcurrentSlab*
SlabHashContext<KeyT, D, ValueT, HashFunc>::getDeviceTablePointer() {
    return d_table_;
}

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
__device__ __forceinline__ uint32_t*
SlabHashContext<KeyT, D, ValueT, HashFunc>::getPointerFromSlab(
        const SlabAddressT& slab_address, const uint32_t lane_id) {
    return slab_list_allocator_ctx_.getPointerFromSlab(slab_address, lane_id);
}

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
__device__ __forceinline__ uint32_t*
SlabHashContext<KeyT, D, ValueT, HashFunc>::getPointerFromBucket(
        const uint32_t bucket_id, const uint32_t lane_id) {
    return reinterpret_cast<uint32_t*>(d_table_) + bucket_id * BASE_UNIT_SIZE +
           lane_id;
}

// this function should be operated in a warp-wide fashion
// TODO: add required asserts to make sure this is true in tests/debugs
template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
__device__ __forceinline__ SlabAllocAddressT
SlabHashContext<KeyT, D, ValueT, HashFunc>::AllocateSlab(
        const uint32_t& lane_id) {
    return slab_list_allocator_ctx_.warpAllocate(lane_id);
}

// a thread-wide function to free the slab that was just allocated
template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
__device__ __forceinline__ void
SlabHashContext<KeyT, D, ValueT, HashFunc>::FreeSlab(
        const SlabAllocAddressT slab_ptr) {
    slab_list_allocator_ctx_.freeUntouched(slab_ptr);
}

//================================================
// Individual Search Unit:
//================================================
template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
__device__ void SlabHashContext<KeyT, D, ValueT, HashFunc>::Search(
        bool& to_search,
        const uint32_t& lane_id,
        const KeyTD& myKey,
        ValueT& myValue,
        const uint32_t bucket_id) {
    uint32_t work_queue = 0;
    uint32_t prev_work_queue = work_queue;
    uint32_t curr_slab_ptr = HEAD_SLAB_POINTER;

    /** > Loop when we have active lanes **/
    while ((work_queue = __ballot_sync(ACTIVE_LANE_MASK, to_search))) {
        /** 0. Restart from linked list head if the last query is finished **/
        curr_slab_ptr = (prev_work_queue != work_queue) ? HEAD_SLAB_POINTER
                                                        : curr_slab_ptr;
        uint32_t src_lane = __ffs(work_queue) - 1;
        uint32_t src_bucket =
                __shfl_sync(ACTIVE_LANE_MASK, bucket_id, src_lane, WARP_WIDTH);

        // TODO generalize it to multiple ints
        KeyTD src_key;
#pragma unroll 1
        for (size_t i = 0; i < D; ++i) {
            src_key[i] = __shfl_sync(ACTIVE_LANE_MASK, myKey[i], src_lane,
                                     WARP_WIDTH);
        }

        /* Each lane in the warp reads a uint in the slab in parallel */
        const uint32_t unit_data =
                (curr_slab_ptr == HEAD_SLAB_POINTER)
                        ? *(getPointerFromBucket(src_bucket, lane_id))
                        : *(getPointerFromSlab(curr_slab_ptr, lane_id));

        int32_t lane_found = laneFoundKeyInWarp(src_key, lane_id, unit_data);

        /** 1. Found in this slab, SUCCEED **/
        if (lane_found >= 0) {
            /* broadcast found value */
            uint32_t found_value = __shfl_sync(ACTIVE_LANE_MASK, unit_data,
                                               lane_found + 1, WARP_WIDTH);

            if (lane_id == src_lane) {
                myValue = value_allocator_ctx_.value_at(found_value);
                to_search = false;
            }
        }

        /** 2. Not found in this slab **/
        else {
            /* broadcast next slab: lane 31 reads 'next' */
            uint32_t curr_slab_next_ptr =
                    __shfl_sync(ACTIVE_LANE_MASK, unit_data,
                                NEXT_SLAB_POINTER_LANE, WARP_WIDTH);

            /** 2.1. Next slab is empty, ABORT **/
            if (curr_slab_next_ptr == EMPTY_SLAB_POINTER) {
                if (lane_id == src_lane) {
                    myValue = static_cast<ValueT>(SEARCH_NOT_FOUND);
                    to_search = false;
                }
            }
            /** 2.2. Next slab exists, RESTART **/
            else {
                curr_slab_ptr = curr_slab_next_ptr;
            }
        }

        prev_work_queue = work_queue;
    }
}

/*
 * each thread inserts a key-value pair into the hash table
 * it is assumed all threads within a warp are present and collaborating with
 * each other with a warp-cooperative work sharing (WCWS) strategy.
 * InsertPair: ABORT if found
 * replacePair: REPLACE if found
 * WE DO NOT ALLOW DUPLICATE KEYS
 */
template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
__device__ void SlabHashContext<KeyT, D, ValueT, HashFunc>::InsertPair(
        bool& to_be_inserted,
        const uint32_t& lane_id,
        const KeyTD& myKey,
        const ValueT& myValue,
        const uint32_t bucket_id) {
    uint32_t work_queue = 0;
    uint32_t prev_work_queue = 0;
    uint32_t curr_slab_ptr = HEAD_SLAB_POINTER;

    /** WARNING: Allocation should be finished in warp,
     * results are unexpected otherwise **/
    int key_addr = EMPTY_KEY;
    int value_addr = EMPTY_KEY;
    if (to_be_inserted) {
        key_addr = key_allocator_ctx_.Malloc();
        key_allocator_ctx_.value_at(key_addr) = myKey;

        value_addr = value_allocator_ctx_.Malloc();
        value_allocator_ctx_.value_at(value_addr) = myValue;
    }

    /** > Loop when we have active lanes **/
    while ((work_queue = __ballot_sync(ACTIVE_LANE_MASK, to_be_inserted))) {
        /** 0. Restart from linked list head if last insertion is finished **/
        curr_slab_ptr = (prev_work_queue != work_queue) ? HEAD_SLAB_POINTER
                                                        : curr_slab_ptr;
        uint32_t src_lane = __ffs(work_queue) - 1;
        uint32_t src_bucket =
                __shfl_sync(ACTIVE_LANE_MASK, bucket_id, src_lane, WARP_WIDTH);
        KeyTD src_key;
#pragma unroll 1
        for (size_t i = 0; i < D; ++i) {
            src_key[i] = __shfl_sync(ACTIVE_LANE_MASK, myKey[i], src_lane,
                                     WARP_WIDTH);
        }

        /* Each lane in the warp reads a uint in the slab */
        uint32_t unit_data =
                (curr_slab_ptr == HEAD_SLAB_POINTER)
                        ? *(getPointerFromBucket(src_bucket, lane_id))
                        : *(getPointerFromSlab(curr_slab_ptr, lane_id));

        int32_t lane_found = laneFoundKeyInWarp(src_key, lane_id, unit_data);
        int32_t lane_empty = laneEmptyKeyInWarp(unit_data);

        /** Branch 1: key already existing, ABORT **/
        if (lane_found >= 0) {
            if (lane_id == src_lane) {
                /* free memory heap */
                to_be_inserted = false;
                key_allocator_ctx_.Free(key_addr);
            }
        }

        /** Branch 2: empty slot available, try to insert **/
        else if (lane_empty >= 0) {
            if (lane_id == src_lane) {
                // TODO: check why we cannot put malloc here
                const uint32_t* p =
                        (curr_slab_ptr == HEAD_SLAB_POINTER)
                                ? getPointerFromBucket(src_bucket, lane_empty)
                                : getPointerFromSlab(curr_slab_ptr, lane_empty);

                uint64_t old_key_value_pair = atomicCAS(
                        (unsigned long long int*)p, EMPTY_PAIR_64,
                        ((uint64_t)(*reinterpret_cast<const uint32_t*>(
                                 reinterpret_cast<const unsigned char*>(
                                         &value_addr)))
                         << 32) |
                                *reinterpret_cast<const uint32_t*>(
                                        reinterpret_cast<const unsigned char*>(
                                                &key_addr)));

                /** Branch 2.1: SUCCEED **/
                if (old_key_value_pair == EMPTY_PAIR_64) {
                    to_be_inserted = false;
                }
                /** Branch 2.2: failed: RESTART
                 *  In the consequent attempt,
                 *  > if the same key was inserted in this slot,
                 *    we fall back to Branch 1;
                 *  > if a different key was inserted,
                 *    we go to Branch 2 or 3.
                 * **/
            }
        }

        /** Branch 3: nothing found in this slab, goto next slab **/
        else {
            /* broadcast next slab */
            uint32_t curr_slab_next_ptr =
                    __shfl_sync(ACTIVE_LANE_MASK, unit_data,
                                NEXT_SLAB_POINTER_LANE, WARP_WIDTH);

            /** Branch 3.1: next slab existing, RESTART this lane **/
            if (curr_slab_next_ptr != EMPTY_SLAB_POINTER) {
                curr_slab_ptr = curr_slab_next_ptr;
            }

            /** Branch 3.2: next slab empty, try to allocate one **/
            else {
                uint32_t new_node_ptr = AllocateSlab(lane_id);

                if (lane_id == NEXT_SLAB_POINTER_LANE) {
                    const uint32_t* p =
                            (curr_slab_ptr == HEAD_SLAB_POINTER)
                                    ? getPointerFromBucket(
                                              src_bucket,
                                              NEXT_SLAB_POINTER_LANE)
                                    : getPointerFromSlab(
                                              curr_slab_ptr,
                                              NEXT_SLAB_POINTER_LANE);

                    uint32_t old_next_slab_ptr = atomicCAS(
                            (unsigned int*)p, EMPTY_SLAB_POINTER, new_node_ptr);

                    /** Branch 3.2.1: other thread allocated, RESTART lane
                     *  In the consequent attempt, goto Branch 2' **/
                    if (old_next_slab_ptr != EMPTY_SLAB_POINTER) {
                        FreeSlab(new_node_ptr);
                    }
                    /** Branch 3.2.2: this thread allocated, RESTART lane,
                     * 'goto Branch 2' **/
                }
            }
        }

        prev_work_queue = work_queue;
    }
}

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
__device__ void SlabHashContext<KeyT, D, ValueT, HashFunc>::Delete(
        bool& to_be_deleted,
        const uint32_t& lane_id,
        const KeyTD& myKey,
        const uint32_t bucket_id) {
    uint32_t work_queue = 0;
    uint32_t prev_work_queue = 0;
    uint32_t curr_slab_ptr = HEAD_SLAB_POINTER;

    /** > Loop when we have active lanes **/
    while ((work_queue = __ballot_sync(ACTIVE_LANE_MASK, to_be_deleted))) {
        /** 0. Restart from linked list head if last insertion is finished
         * **/
        curr_slab_ptr = (prev_work_queue != work_queue) ? HEAD_SLAB_POINTER
                                                        : curr_slab_ptr;
        uint32_t src_lane = __ffs(work_queue) - 1;
        uint32_t src_bucket =
                __shfl_sync(ACTIVE_LANE_MASK, bucket_id, src_lane, WARP_WIDTH);

        KeyTD src_key;
#pragma unroll 1
        for (size_t i = 0; i < D; ++i) {
            src_key[i] = __shfl_sync(ACTIVE_LANE_MASK, myKey[i], src_lane,
                                     WARP_WIDTH);
        }

        const uint32_t unit_data =
                (curr_slab_ptr == HEAD_SLAB_POINTER)
                        ? *(getPointerFromBucket(src_bucket, lane_id))
                        : *(getPointerFromSlab(curr_slab_ptr, lane_id));

        int32_t lane_found = laneFoundKeyInWarp(src_key, lane_id, unit_data);

        /** Branch 1: key found **/
        if (lane_found >= 0) {
            uint32_t src_addr = __shfl_sync(ACTIVE_LANE_MASK, unit_data,
                                            lane_found, WARP_WIDTH);
            uint32_t src_value_addr = __shfl_sync(ACTIVE_LANE_MASK, unit_data,
                                                  lane_found + 1, WARP_WIDTH);
            if (lane_id == src_lane) {
                uint32_t* p =
                        (curr_slab_ptr == HEAD_SLAB_POINTER)
                                ? getPointerFromBucket(src_bucket, lane_found)
                                : getPointerFromSlab(curr_slab_ptr, lane_found);
                uint64_t key_value_pair_to_delete =
                        *reinterpret_cast<unsigned long long int*>(p);

                // TODO: keep in mind the potential double free problem
                uint64_t old_key_value_pair =
                        atomicCAS(reinterpret_cast<unsigned long long int*>(p),
                                  key_value_pair_to_delete, EMPTY_PAIR_64);
                /** Branch 1.1: this thread reset, free src_addr **/
                if (old_key_value_pair == key_value_pair_to_delete) {
                    key_allocator_ctx_.Free(src_addr);
                    value_allocator_ctx_.Free(src_value_addr);
                }
                /** Branch 1.2: other thread did the job, avoid double free
                 * **/
                to_be_deleted = false;
            }
        } else {  // no matching slot found:
            uint32_t curr_slab_next_ptr =
                    __shfl_sync(ACTIVE_LANE_MASK, unit_data,
                                NEXT_SLAB_POINTER_LANE, WARP_WIDTH);
            if (curr_slab_next_ptr == EMPTY_SLAB_POINTER) {
                // not found:
                to_be_deleted = false;
            } else {
                curr_slab_ptr = curr_slab_next_ptr;
            }
        }
        prev_work_queue = work_queue;
    }
}
