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

//================================================
// Individual Search Unit:
//================================================
template <typename KeyT, typename ValueT>
__device__ void GpuSlabHashContext<KeyT, ValueT>::searchKey(
        bool& to_search,
        const uint32_t& lane_id,
        const KeyT& myKey,
        ValueT& myValue,
        const uint32_t bucket_id) {
    uint32_t work_queue = 0;
    uint32_t prev_work_queue = work_queue;
    uint32_t curr_slab_ptr = A_INDEX_POINTER;

    /** > Loop when we have active lanes **/
    while ((work_queue = __ballot_sync(0xFFFFFFFF, to_search))) {
        /** 0. Restart from linked list head if last lane is finished **/
        curr_slab_ptr = (prev_work_queue != work_queue) ? A_INDEX_POINTER
                                                        : curr_slab_ptr;
        uint32_t src_lane = __ffs(work_queue) - 1;
        uint32_t src_bucket = __shfl_sync(0xFFFFFFFF, bucket_id, src_lane, 32);
        uint32_t src_key = __shfl_sync(0xFFFFFFFF, myKey, src_lane, 32);

        /* Each lane reads a uint in the slab; lane 31 reads 'next' */
        const uint32_t src_unit_data =
                (curr_slab_ptr == A_INDEX_POINTER)
                        ? *(getPointerFromBucket(src_bucket, lane_id))
                        : *(getPointerFromSlab(curr_slab_ptr, lane_id));

        int32_t lane_found =
                SlabHash_NS::findKeyPerWarp<KeyT>(src_key, src_unit_data);

        /** 1. Found in this slab **/
        if (lane_found >= 0) {
            /* broadcast found value */
            uint32_t found_value =
                    __shfl_sync(0xFFFFFFFF, src_unit_data, lane_found + 1, 32);

            if (lane_id == src_lane) {
                myValue = *reinterpret_cast<const ValueT*>(
                        reinterpret_cast<const unsigned char*>(&found_value));
                to_search = false;
            }
        }

        /** 2. Not found in this slab **/
        else {
            /* broadcast next slab */
            uint32_t curr_slab_next_ptr =
                    __shfl_sync(0xFFFFFFFF, src_unit_data, 31, 32);

            /** 2.1. Next slab is empty, abort **/
            if (curr_slab_next_ptr == EMPTY_INDEX_POINTER) {
                if (lane_id == src_lane) {
                    myValue = static_cast<ValueT>(SEARCH_NOT_FOUND);
                    to_search = false;
                }
            }
            /** 2.2. Next slab exists **/
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
 * insertPair: ABORT if found
 * replacePair: REPLACE if found
 * WE DO NOT ALLOW DUPLICATE KEYS
 */
template <typename KeyT, typename ValueT>
__device__ void GpuSlabHashContext<KeyT, ValueT>::insertPair(
        bool& to_be_inserted,
        const uint32_t& lane_id,
        const KeyT& myKey,
        const ValueT& myValue,
        const uint32_t bucket_id) {
    uint32_t work_queue = 0;
    uint32_t prev_work_queue = 0;
    uint32_t curr_slab_ptr = A_INDEX_POINTER;

    /** > Loop when we have active lanes **/
    while ((work_queue = __ballot_sync(0xFFFFFFFF, to_be_inserted))) {
        /** 0. Restart from linked list head if last lane is finished **/
        curr_slab_ptr = (prev_work_queue != work_queue) ? A_INDEX_POINTER
                                                        : curr_slab_ptr;
        uint32_t src_lane = __ffs(work_queue) - 1;
        uint32_t src_bucket = __shfl_sync(0xFFFFFFFF, bucket_id, src_lane, 32);
        uint32_t src_key = __shfl_sync(0xFFFFFFFF, myKey, src_lane, 32);

        /* Each lane reads a uint in the slab; lane 31 reads 'next' */
        uint32_t src_unit_data =
                (curr_slab_ptr == A_INDEX_POINTER)
                        ? *(getPointerFromBucket(src_bucket, lane_id))
                        : *(getPointerFromSlab(curr_slab_ptr, lane_id));
        uint64_t old_key_value_pair = 0;

        int32_t lane_found =
                SlabHash_NS::findKeyPerWarp<KeyT>(src_key, src_unit_data);
        int32_t lane_empty = SlabHash_NS::findEmptyPerWarp<KeyT>(src_unit_data);

        /** Branch 1: key already existing, ABORT **/
        if (lane_found >= 0) {
            if (lane_id == src_lane) {
                /* free memory heap */
                to_be_inserted = false;
            }
        }

        /** Branch 2: empty slot available, try to insert **/
        else if (lane_empty >= 0) {
            if (lane_id == src_lane) {
                const uint32_t* p =
                        (curr_slab_ptr == A_INDEX_POINTER)
                                ? getPointerFromBucket(src_bucket, lane_empty)
                                : getPointerFromSlab(curr_slab_ptr, lane_empty);

                old_key_value_pair = atomicCAS(
                        (unsigned long long int*)p, EMPTY_PAIR_64,
                        ((uint64_t)(*reinterpret_cast<const uint32_t*>(
                                 reinterpret_cast<const unsigned char*>(
                                         &myValue)))
                         << 32) |
                                *reinterpret_cast<const uint32_t*>(
                                        reinterpret_cast<const unsigned char*>(
                                                &myKey)));

                /** Branch 2.1: insertion succeeded **/
                if (old_key_value_pair == EMPTY_PAIR_64) {
                    to_be_inserted = false;
                    // printf("inserted!\n");
                }
                /** Branch 2.2: failed: RESTART lane
                 *  In the following attempt,
                 *  > if the same key was inserted in this slot,
                 *    we fall back to Branch 1;
                 *  > if a diff key was inserted,
                 *    we go to Branch 2 or 3.
                 * **/
            }
        }

        /** Branch 3: nothing found in this slab, goto next slab **/
        else {
            /* broadcast next slab */
            uint32_t curr_slab_ptr_ptr =
                    __shfl_sync(0xFFFFFFFF, src_unit_data, 31, 32);

            /** Branch 3.1: next slab existing, restart this lane **/
            if (curr_slab_ptr_ptr != EMPTY_INDEX_POINTER) {
                curr_slab_ptr = curr_slab_ptr_ptr;
            }

            /** Branch 3.2: next slab empty, try to allocate one **/
            else {
                uint32_t new_node_ptr = allocateSlab(lane_id);

                if (lane_id == 31) {
                    const uint32_t* p =
                            (curr_slab_ptr == A_INDEX_POINTER)
                                    ? getPointerFromBucket(src_bucket, 31)
                                    : getPointerFromSlab(curr_slab_ptr, 31);

                    uint32_t old_next_slab_ptr =
                            atomicCAS((unsigned int*)p, EMPTY_INDEX_POINTER,
                                      new_node_ptr);

                    /** Branch 3.2.1: other thread allocated, RESTART lane
                     *  Same as 'goto Branch 2', but don't duplicate code. **/
                    if (old_next_slab_ptr != EMPTY_INDEX_POINTER) {
                        freeSlab(new_node_ptr);
                    }
                    // printf("slab allocated!\n");
                    /** Branch 3.2.2: similar, RESTART lane, 'goto Branch 2' **/
                }  // lane 31
            }      // next slab empty
        }          // next slab

        prev_work_queue = work_queue;
    }
}

template <typename KeyT, typename ValueT>
__device__ void GpuSlabHashContext<KeyT, ValueT>::deleteKey(
        bool& to_be_deleted,
        const uint32_t& lane_id,
        const KeyT& myKey,
        const uint32_t bucket_id) {
    uint32_t work_queue = 0;
    uint32_t prev_work_queue = 0;
    uint32_t curr_slab_ptr = A_INDEX_POINTER;

    while ((work_queue = __ballot_sync(0xFFFFFFFF, to_be_deleted))) {
        // to know whether it is a base node, or a regular node
        curr_slab_ptr =
                (prev_work_queue != work_queue)
                        ? A_INDEX_POINTER
                        : curr_slab_ptr;  // a successfull insertion in the warp
        uint32_t src_lane = __ffs(work_queue) - 1;
        uint32_t src_key = __shfl_sync(
                0xFFFFFFFF,
                *reinterpret_cast<const uint32_t*>(
                        reinterpret_cast<const unsigned char*>(&myKey)),
                src_lane, 32);
        uint32_t src_bucket = __shfl_sync(0xFFFFFFFF, bucket_id, src_lane, 32);
        // starting with a base node OR regular node:
        // need to define different masks to extract super block index, memory
        // block index, and the memory unit index

        const uint32_t src_unit_data =
                (curr_slab_ptr == A_INDEX_POINTER)
                        ? *(getPointerFromBucket(src_bucket, lane_id))
                        : *(getPointerFromSlab(curr_slab_ptr, lane_id));

        // looking for the item to be deleted:
        uint32_t isFound =
                (__ballot_sync(0xFFFFFFFF, src_unit_data == src_key)) &
                REGULAR_NODE_KEY_MASK;

        if (isFound == 0) {  // no matching slot found:
            uint32_t curr_slab_ptr_ptr =
                    __shfl_sync(0xFFFFFFFF, src_unit_data, 31, 32);
            if (curr_slab_ptr_ptr == EMPTY_INDEX_POINTER) {
                // not found:
                to_be_deleted = false;
            } else {
                curr_slab_ptr = curr_slab_ptr_ptr;
            }
        } else {  // The wanted key found:
            int dest_lane = __ffs(isFound & REGULAR_NODE_KEY_MASK) - 1;
            if (lane_id == src_lane) {
                uint32_t* p =
                        (curr_slab_ptr == A_INDEX_POINTER)
                                ? getPointerFromBucket(src_bucket, dest_lane)
                                : getPointerFromSlab(curr_slab_ptr, dest_lane);
                // deleting that item (no atomics)
                *(reinterpret_cast<uint64_t*>(p)) = EMPTY_PAIR_64;
                to_be_deleted = false;
            }
        }
        prev_work_queue = work_queue;
    }
}
