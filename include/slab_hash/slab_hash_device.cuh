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

// fixed known parameters:
template <typename KeyT, typename ValueT, typename HashFunc>
SlabHashContext<KeyT, ValueT, HashFunc>::SlabHashContext()
    : num_buckets_(0), bucket_list_head_(nullptr) {
    // a single slab on a ConcurrentMap should be 128 bytes
    static_assert(sizeof(Slab) == (WARP_WIDTH * sizeof(uint32_t)));
}

template <typename KeyT, typename ValueT, typename HashFunc>
__host__ void SlabHashContext<KeyT, ValueT, HashFunc>::Setup(
        Slab* bucket_list_head,
        const uint32_t num_buckets,
        const SlabAllocContext& allocator_ctx,
        const MemoryAllocContext<thrust::pair<KeyT, ValueT>>&
                pair_allocator_ctx) {
    bucket_list_head_ = bucket_list_head;

    num_buckets_ = num_buckets;
    slab_list_allocator_ctx_ = allocator_ctx;
    pair_allocator_ctx_ = pair_allocator_ctx;
}

template <typename KeyT, typename ValueT, typename HashFunc>
__device__ __host__ __forceinline__ uint32_t
SlabHashContext<KeyT, ValueT, HashFunc>::ComputeBucket(const KeyT& key) const {
    return hash_fn_(key) % num_buckets_;
}

template <typename KeyT, typename ValueT, typename HashFunc>
__device__ __forceinline__ void
SlabHashContext<KeyT, ValueT, HashFunc>::WarpSyncKey(const KeyT& key,
                                                     const uint32_t lane_id,
                                                     KeyT& ret) {
    const int chunks = sizeof(KeyT) / sizeof(int);
#pragma unroll 1
    for (size_t i = 0; i < chunks; ++i) {
        ((int*)(&ret))[i] = __shfl_sync(ACTIVE_LANE_MASK, ((int*)(&key))[i],
                                        lane_id, WARP_WIDTH);
    }
}

template <typename KeyT, typename ValueT, typename HashFunc>
__device__ int32_t SlabHashContext<KeyT, ValueT, HashFunc>::WarpFindKey(
        const KeyT& src_key, const uint32_t lane_id, const uint32_t unit_data) {
    bool is_lane_found =
            /* select key lanes */
            ((1 << lane_id) & REGULAR_NODE_KEY_MASK)
            /* validate key addrs */
            && (unit_data != EMPTY_KEY)
            /* find keys in memory heap */
            && pair_allocator_ctx_.value_at(unit_data).first == src_key;

    return __ffs(__ballot_sync(REGULAR_NODE_KEY_MASK, is_lane_found)) - 1;
}

template <typename KeyT, typename ValueT, typename HashFunc>
__device__ __forceinline__ int32_t
SlabHashContext<KeyT, ValueT, HashFunc>::WarpFindEmpty(
        const uint32_t unit_data) {
    bool is_lane_empty = (unit_data == EMPTY_KEY);

    return __ffs(__ballot_sync(REGULAR_NODE_KEY_MASK, is_lane_empty)) - 1;
}

template <typename KeyT, typename ValueT, typename HashFunc>
__device__ __forceinline__ slab_ptr_t
SlabHashContext<KeyT, ValueT, HashFunc>::AllocateSlab(const uint32_t lane_id) {
    return slab_list_allocator_ctx_.WarpAllocate(lane_id);
}

template <typename KeyT, typename ValueT, typename HashFunc>
__device__ __forceinline__ void
SlabHashContext<KeyT, ValueT, HashFunc>::FreeSlab(const slab_ptr_t slab_ptr) {
    slab_list_allocator_ctx_.FreeUntouched(slab_ptr);
}

//================================================
// Individual Search Unit:
//================================================
template <typename KeyT, typename ValueT, typename HashFunc>
__device__ void SlabHashContext<KeyT, ValueT, HashFunc>::Search(
        bool& to_search,
        const uint32_t lane_id,
        const uint32_t bucket_id,
        const KeyT& query_key,
        ValueT& found_value,
        uint8_t& found) {
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

        KeyT src_key;
        WarpSyncKey(query_key, src_lane, src_key);

        /* Each lane in the warp reads a uint in the slab in parallel */
        const uint32_t unit_data =
                (curr_slab_ptr == HEAD_SLAB_POINTER)
                        ? *(get_unit_ptr_from_list_head(src_bucket, lane_id))
                        : *(get_unit_ptr_from_list_nodes(curr_slab_ptr,
                                                         lane_id));

        int32_t lane_found = WarpFindKey(src_key, lane_id, unit_data);

        /** 1. Found in this slab, SUCCEED **/
        if (lane_found >= 0) {
            /* broadcast found value */
            internal_ptr_t found_pair_internal_ptr = __shfl_sync(
                    ACTIVE_LANE_MASK, unit_data, lane_found, WARP_WIDTH);

            if (lane_id == src_lane) {
                found_value =
                        pair_allocator_ctx_.value_at(found_pair_internal_ptr)
                                .second;
                found = 1;
                to_search = false;
            }
        }

        /** 2. Not found in this slab **/
        else {
            /* broadcast next slab: lane 31 reads 'next' */
            slab_ptr_t next_slab_ptr =
                    __shfl_sync(ACTIVE_LANE_MASK, unit_data,
                                NEXT_SLAB_POINTER_LANE, WARP_WIDTH);

            /** 2.1. Next slab is empty, ABORT **/
            if (next_slab_ptr == EMPTY_SLAB_POINTER) {
                if (lane_id == src_lane) {
                    found = 0;
                    to_search = false;
                }
            }
            /** 2.2. Next slab exists, RESTART **/
            else {
                curr_slab_ptr = next_slab_ptr;
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
template <typename KeyT, typename ValueT, typename HashFunc>
__device__ void SlabHashContext<KeyT, ValueT, HashFunc>::InsertPair(
        bool& to_be_inserted,
        const uint32_t lane_id,
        const uint32_t bucket_id,
        const KeyT& key,
        const ValueT& value) {
    uint32_t work_queue = 0;
    uint32_t prev_work_queue = 0;
    uint32_t curr_slab_ptr = HEAD_SLAB_POINTER;

    /** WARNING: Allocation should be finished in warp,
     * results are unexpected otherwise **/
    // int prealloc_key_internal_ptr = EMPTY_KEY;
    // int prealloc_value_internal_ptr = EMPTY_KEY;
    int prealloc_pair_internal_ptr = EMPTY_KEY;
    if (to_be_inserted) {
        prealloc_pair_internal_ptr = pair_allocator_ctx_.Allocate();
        pair_allocator_ctx_.value_at(prealloc_pair_internal_ptr) =
                thrust::make_pair(key, value);
        // prealloc_key_internal_ptr = key_allocator_ctx_.Allocate();
        // key_allocator_ctx_.value_at(prealloc_key_internal_ptr) = key;

        // prealloc_value_internal_ptr = value_allocator_ctx_.Allocate();
        // value_allocator_ctx_.value_at(prealloc_value_internal_ptr) = value;
    }

    /** > Loop when we have active lanes **/
    while ((work_queue = __ballot_sync(ACTIVE_LANE_MASK, to_be_inserted))) {
        /** 0. Restart from linked list head if last insertion is finished **/
        curr_slab_ptr = (prev_work_queue != work_queue) ? HEAD_SLAB_POINTER
                                                        : curr_slab_ptr;
        uint32_t src_lane = __ffs(work_queue) - 1;
        uint32_t src_bucket =
                __shfl_sync(ACTIVE_LANE_MASK, bucket_id, src_lane, WARP_WIDTH);
        KeyT src_key;
        WarpSyncKey(key, src_lane, src_key);

        /* Each lane in the warp reads a uint in the slab */
        uint32_t unit_data =
                (curr_slab_ptr == HEAD_SLAB_POINTER)
                        ? *(get_unit_ptr_from_list_head(src_bucket, lane_id))
                        : *(get_unit_ptr_from_list_nodes(curr_slab_ptr,
                                                         lane_id));

        int32_t lane_found = WarpFindKey(src_key, lane_id, unit_data);
        int32_t lane_empty = WarpFindEmpty(unit_data);

        /** Branch 1: key already existing, ABORT **/
        if (lane_found >= 0) {
            if (lane_id == src_lane) {
                /* free memory heap */
                to_be_inserted = false;
                pair_allocator_ctx_.Free(prealloc_pair_internal_ptr);
                // key_allocator_ctx_.Free(prealloc_key_internal_ptr);
                // value_allocator_ctx_.Free(prealloc_value_internal_ptr);
            }
        }

        /** Branch 2: empty slot available, try to insert **/
        else if (lane_empty >= 0) {
            if (lane_id == src_lane) {
                // TODO: check why we cannot put malloc here
                const uint32_t* unit_data_ptr =
                        (curr_slab_ptr == HEAD_SLAB_POINTER)
                                ? get_unit_ptr_from_list_head(src_bucket,
                                                              lane_empty)
                                : get_unit_ptr_from_list_nodes(curr_slab_ptr,
                                                               lane_empty);
                internal_ptr_t old_pair_internal_ptr =
                        atomicCAS((unsigned int*)unit_data_ptr, EMPTY_KEY,
                                  prealloc_pair_internal_ptr);

                // paired_internal_ptr_t new_key_value_ptr_pair =
                // MAKE_PAIRED_PTR(
                //         prealloc_key_internal_ptr,
                //         prealloc_value_internal_ptr);
                // paired_internal_ptr_t old_key_value_ptr_pair = atomicCAS(
                //         (unsigned long long int*)unit_data_ptr,
                //         EMPTY_PAIR_64,
                //         (*(unsigned long long int*)&new_key_value_ptr_pair));

                /** Branch 2.1: SUCCEED **/
                if (old_pair_internal_ptr == EMPTY_KEY) {
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
            slab_ptr_t next_slab_ptr =
                    __shfl_sync(ACTIVE_LANE_MASK, unit_data,
                                NEXT_SLAB_POINTER_LANE, WARP_WIDTH);

            /** Branch 3.1: next slab existing, RESTART this lane **/
            if (next_slab_ptr != EMPTY_SLAB_POINTER) {
                curr_slab_ptr = next_slab_ptr;
            }

            /** Branch 3.2: next slab empty, try to allocate one **/
            else {
                slab_ptr_t new_next_slab_ptr = AllocateSlab(lane_id);

                if (lane_id == NEXT_SLAB_POINTER_LANE) {
                    const uint32_t* unit_data_ptr =
                            (curr_slab_ptr == HEAD_SLAB_POINTER)
                                    ? get_unit_ptr_from_list_head(
                                              src_bucket,
                                              NEXT_SLAB_POINTER_LANE)
                                    : get_unit_ptr_from_list_nodes(
                                              curr_slab_ptr,
                                              NEXT_SLAB_POINTER_LANE);

                    slab_ptr_t old_next_slab_ptr =
                            atomicCAS((unsigned int*)unit_data_ptr,
                                      EMPTY_SLAB_POINTER, new_next_slab_ptr);

                    /** Branch 3.2.1: other thread allocated, RESTART lane
                     *  In the consequent attempt, goto Branch 2' **/
                    if (old_next_slab_ptr != EMPTY_SLAB_POINTER) {
                        FreeSlab(new_next_slab_ptr);
                    }
                    /** Branch 3.2.2: this thread allocated, RESTART lane,
                     * 'goto Branch 2' **/
                }
            }
        }

        prev_work_queue = work_queue;
    }
}

template <typename KeyT, typename ValueT, typename HashFunc>
__device__ void SlabHashContext<KeyT, ValueT, HashFunc>::Delete(
        bool& to_be_deleted,
        const uint32_t lane_id,
        const uint32_t bucket_id,
        const KeyT& key) {
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

        KeyT src_key;
        WarpSyncKey(key, src_lane, src_key);

        const uint32_t unit_data =
                (curr_slab_ptr == HEAD_SLAB_POINTER)
                        ? *(get_unit_ptr_from_list_head(src_bucket, lane_id))
                        : *(get_unit_ptr_from_list_nodes(curr_slab_ptr,
                                                         lane_id));

        int32_t lane_found = WarpFindKey(src_key, lane_id, unit_data);

        /** Branch 1: key found **/
        if (lane_found >= 0) {
            internal_ptr_t src_pair_internal_ptr = __shfl_sync(
                    ACTIVE_LANE_MASK, unit_data, lane_found, WARP_WIDTH);
            // internal_ptr_t src_key_internal_ptr = __shfl_sync(
            //         ACTIVE_LANE_MASK, unit_data, lane_found, WARP_WIDTH);
            // internal_ptr_t src_value_internal_ptr = __shfl_sync(
            //         ACTIVE_LANE_MASK, unit_data, lane_found + 1, WARP_WIDTH);

            if (lane_id == src_lane) {
                uint32_t* unit_data_ptr =
                        (curr_slab_ptr == HEAD_SLAB_POINTER)
                                ? get_unit_ptr_from_list_head(src_bucket,
                                                              lane_found)
                                : get_unit_ptr_from_list_nodes(curr_slab_ptr,
                                                               lane_found);
                internal_ptr_t pair_to_delete = *unit_data_ptr;

                // TODO: keep in mind the potential double free problem
                internal_ptr_t old_key_value_pair =
                        atomicCAS((unsigned int*)(unit_data_ptr),
                                  pair_to_delete, EMPTY_KEY);
                /** Branch 1.1: this thread reset, free src_addr **/
                if (old_key_value_pair == pair_to_delete) {
                    pair_allocator_ctx_.Free(src_pair_internal_ptr);
                    // key_allocator_ctx_.Free(src_key_internal_ptr);
                    // value_allocator_ctx_.Free(src_value_internal_ptr);
                }
                /** Branch 1.2: other thread did the job, avoid double free
                 * **/
                to_be_deleted = false;
            }
        } else {  // no matching slot found:
            slab_ptr_t next_slab_ptr =
                    __shfl_sync(ACTIVE_LANE_MASK, unit_data,
                                NEXT_SLAB_POINTER_LANE, WARP_WIDTH);
            if (next_slab_ptr == EMPTY_SLAB_POINTER) {
                // not found:
                to_be_deleted = false;
            } else {
                curr_slab_ptr = next_slab_ptr;
            }
        }
        prev_work_queue = work_queue;
    }
}
