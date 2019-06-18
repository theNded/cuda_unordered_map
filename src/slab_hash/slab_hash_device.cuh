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
        bool& to_be_searched,
        const uint32_t& laneId,
        const KeyT& myKey,
        ValueT& myValue,
        const uint32_t bucket_id) {
    using SlabHashT = ConcurrentSlab;
    uint32_t work_queue = 0;
    uint32_t last_work_queue = work_queue;
    uint32_t next = A_INDEX_POINTER;

    while ((work_queue = __ballot_sync(0xFFFFFFFF, to_be_searched))) {
        next = (last_work_queue != work_queue)
                       ? A_INDEX_POINTER
                       : next;  // a successfull insertion in the warp
        uint32_t src_lane = __ffs(work_queue) - 1;
        uint32_t src_bucket = __shfl_sync(0xFFFFFFFF, bucket_id, src_lane, 32);
        uint32_t wanted_key = __shfl_sync(
                0xFFFFFFFF,
                *reinterpret_cast<const uint32_t*>(
                        reinterpret_cast<const unsigned char*>(&myKey)),
                src_lane, 32);
        const uint32_t src_unit_data =
                (next == A_INDEX_POINTER)
                        ? *(getPointerFromBucket(src_bucket, laneId))
                        : *(getPointerFromSlab(next, laneId));

        int32_t found_lane =
                SlabHash_NS::findKeyPerWarp<KeyT>(wanted_key, src_unit_data);

        if (found_lane < 0) {  // not found
            uint32_t next_ptr = __shfl_sync(0xFFFFFFFF, src_unit_data, 31, 32);
            if (next_ptr == EMPTY_INDEX_POINTER) {  // not found
                if (laneId == src_lane) {
                    myValue = static_cast<ValueT>(SEARCH_NOT_FOUND);
                    to_be_searched = false;
                }
            } else {
                next = next_ptr;
            }
        } else {  // found the key:
            uint32_t found_value =
                    __shfl_sync(0xFFFFFFFF, src_unit_data, found_lane + 1, 32);
            if (laneId == src_lane) {
                myValue = *reinterpret_cast<const ValueT*>(
                        reinterpret_cast<const unsigned char*>(&found_value));
                to_be_searched = false;
            }
        }
        last_work_queue = work_queue;
    }
}

/*
 * each thread inserts a key-value pair into the hash table
 * it is assumed all threads within a warp are present and collaborating with
 * each other with a warp-cooperative work sharing (WCWS) strategy.
 */
template <typename KeyT, typename ValueT>
__device__ void GpuSlabHashContext<KeyT, ValueT>::insertPair(
        bool& to_be_inserted,
        const uint32_t& laneId,
        const KeyT& myKey,
        const ValueT& myValue,
        const uint32_t bucket_id) {
    using SlabHashT = ConcurrentSlab;
    uint32_t work_queue = 0;
    uint32_t last_work_queue = 0;
    uint32_t next = A_INDEX_POINTER;

    while ((work_queue = __ballot_sync(0xFFFFFFFF, to_be_inserted))) {
        // to know whether it is a base node, or a regular node
        next = (last_work_queue != work_queue)
                       ? A_INDEX_POINTER
                       : next;  // a successfull insertion in the warp
        uint32_t src_lane = __ffs(work_queue) - 1;
        uint32_t src_bucket = __shfl_sync(0xFFFFFFFF, bucket_id, src_lane, 32);

        uint32_t src_unit_data =
                (next == A_INDEX_POINTER)
                        ? *(getPointerFromBucket(src_bucket, laneId))
                        : *(getPointerFromSlab(next, laneId));
        uint64_t old_key_value_pair = 0;

        uint32_t isEmpty =
                (__ballot_sync(0xFFFFFFFF, src_unit_data == EMPTY_KEY)) &
                REGULAR_NODE_KEY_MASK;
        if (isEmpty == 0) {  // no empty slot available:
            uint32_t next_ptr = __shfl_sync(0xFFFFFFFF, src_unit_data, 31, 32);
            if (next_ptr == EMPTY_INDEX_POINTER) {
                // allocate a new node:
                uint32_t new_node_ptr = allocateSlab(laneId);

                // TODO: experiment if it's better to use lane 0 instead
                if (laneId == 31) {
                    const uint32_t* p =
                            (next == A_INDEX_POINTER)
                                    ? getPointerFromBucket(src_bucket, 31)
                                    : getPointerFromSlab(next, 31);

                    uint32_t temp =
                            atomicCAS((unsigned int*)p, EMPTY_INDEX_POINTER,
                                      new_node_ptr);
                    // check whether it was successful, and
                    // free the allocated memory otherwise
                    if (temp != EMPTY_INDEX_POINTER) {
                        freeSlab(new_node_ptr);
                    }
                }
            } else {
                next = next_ptr;
            }
        } else {  // there is an empty slot available
            int dest_lane = __ffs(isEmpty & REGULAR_NODE_KEY_MASK) - 1;
            if (laneId == src_lane) {
                const uint32_t* p =
                        (next == A_INDEX_POINTER)
                                ? getPointerFromBucket(src_bucket, dest_lane)
                                : getPointerFromSlab(next, dest_lane);

                old_key_value_pair = atomicCAS(
                        (unsigned long long int*)p, EMPTY_PAIR_64,
                        ((uint64_t)(*reinterpret_cast<const uint32_t*>(
                                 reinterpret_cast<const unsigned char*>(
                                         &myValue)))
                         << 32) |
                                *reinterpret_cast<const uint32_t*>(
                                        reinterpret_cast<const unsigned char*>(
                                                &myKey)));
                if (old_key_value_pair == EMPTY_PAIR_64)
                    to_be_inserted = false;  // succesfful insertion
            }
        }
        last_work_queue = work_queue;
    }
}

template <typename KeyT, typename ValueT>
__device__ void GpuSlabHashContext<KeyT, ValueT>::deleteKey(
        bool& to_be_deleted,
        const uint32_t& laneId,
        const KeyT& myKey,
        const uint32_t bucket_id) {
    // delete all instances of key

    uint32_t work_queue = 0;
    uint32_t last_work_queue = 0;
    uint32_t next = A_INDEX_POINTER;

    while ((work_queue = __ballot_sync(0xFFFFFFFF, to_be_deleted))) {
        // to know whether it is a base node, or a regular node
        next = (last_work_queue != work_queue)
                       ? A_INDEX_POINTER
                       : next;  // a successfull insertion in the warp
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
                (next == A_INDEX_POINTER)
                        ? *(getPointerFromBucket(src_bucket, laneId))
                        : *(getPointerFromSlab(next, laneId));

        // looking for the item to be deleted:
        uint32_t isFound =
                (__ballot_sync(0xFFFFFFFF, src_unit_data == src_key)) &
                REGULAR_NODE_KEY_MASK;

        if (isFound == 0) {  // no matching slot found:
            uint32_t next_ptr = __shfl_sync(0xFFFFFFFF, src_unit_data, 31, 32);
            if (next_ptr == EMPTY_INDEX_POINTER) {
                // not found:
                to_be_deleted = false;
            } else {
                next = next_ptr;
            }
        } else {  // The wanted key found:
            int dest_lane = __ffs(isFound & REGULAR_NODE_KEY_MASK) - 1;
            if (laneId == src_lane) {
                uint32_t* p =
                        (next == A_INDEX_POINTER)
                                ? getPointerFromBucket(src_bucket, dest_lane)
                                : getPointerFromSlab(next, dest_lane);
                // deleting that item (no atomics)
                *(reinterpret_cast<uint64_t*>(p)) = EMPTY_PAIR_64;
                to_be_deleted = false;
            }
        }
        last_work_queue = work_queue;
    }
}