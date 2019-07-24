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
#include "slab_hash_device.cuh"

//=== Individual search kernel:
template <typename KeyT, typename ValueT, typename HashFunc>
__global__ void SearchKernel(
        SlabHashContext<KeyT, ValueT, HashFunc> slab_hash_ctx,
        KeyT* keys,
        ValueT* values,
        uint8_t* founds,
        uint32_t num_queries) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    /* This warp is idle */
    if ((tid - lane_id) >= num_queries) {
        return;
    }

    /* Initialize the memory allocator on each warp */
    slab_hash_ctx.getAllocatorContext().initAllocator(tid, lane_id);

    bool lane_active = false;
    uint32_t bucket_id = 0;
    KeyT key;
    ValueT value;
    uint8_t found;

    if (tid < num_queries) {
        lane_active = true;
        key = keys[tid];
        bucket_id = slab_hash_ctx.ComputeBucket(key);
    }

    slab_hash_ctx.Search(lane_active, lane_id, bucket_id, key, value, found);

    if (tid < num_queries) {
        values[tid] = value;
        founds[tid] = found;
    }
}

template <typename KeyT, typename ValueT, typename HashFunc>
__global__ void InsertKernel(
        SlabHashContext<KeyT, ValueT, HashFunc> slab_hash_ctx,
        KeyT* keys,
        ValueT* values,
        uint32_t num_keys) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    if ((tid - lane_id) >= num_keys) {
        return;
    }

    slab_hash_ctx.getAllocatorContext().initAllocator(tid, lane_id);

    bool lane_active = false;
    uint32_t bucket_id = 0;
    KeyT key;
    ValueT value;

    if (tid < num_keys) {
        lane_active = true;
        key = keys[tid];
        value = values[tid];
        bucket_id = slab_hash_ctx.ComputeBucket(key);
    }

    slab_hash_ctx.InsertPair(lane_active, lane_id, bucket_id, key, value);
}

template <typename KeyT, typename ValueT, typename HashFunc>
__global__ void DeleteKernel(
        SlabHashContext<KeyT, ValueT, HashFunc> slab_hash_ctx,
        KeyT* keys,
        uint32_t num_keys) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    if ((tid - lane_id) >= num_keys) {
        return;
    }

    slab_hash_ctx.getAllocatorContext().initAllocator(tid, lane_id);

    bool lane_active = false;
    uint32_t bucket_id = 0;
    KeyT key;

    if (tid < num_keys) {
        lane_active = true;
        key = keys[tid];
        bucket_id = slab_hash_ctx.ComputeBucket(key);
    }

    slab_hash_ctx.Delete(lane_active, lane_id, bucket_id, key);
}

/*
 * This kernel can be used to compute total number of elements within each
 * bucket. The final results per bucket is stored in d_count_result array
 */
template <typename KeyT, typename ValueT, typename HashFunc>
__global__ void bucket_count_kernel(
        SlabHashContext<KeyT, ValueT, HashFunc> slab_hash_ctx,
        uint32_t* d_count_result,
        uint32_t num_buckets) {
    // global warp ID
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t wid = tid >> 5;
    // assigning a warp per bucket
    if (wid >= num_buckets) {
        return;
    }

    uint32_t lane_id = threadIdx.x & 0x1F;

    // initializing the memory allocator on each warp:
    slab_hash_ctx.getAllocatorContext().initAllocator(tid, lane_id);

    uint32_t count = 0;

    uint32_t src_unit_data =
            *slab_hash_ctx.get_unit_ptr_from_list_head(wid, lane_id);

    count += __popc(
            __ballot_sync(REGULAR_NODE_KEY_MASK, src_unit_data != EMPTY_KEY));
    uint32_t next = __shfl_sync(0xFFFFFFFF, src_unit_data, 31, 32);

    while (next != EMPTY_SLAB_POINTER) {
        src_unit_data =
                *slab_hash_ctx.get_unit_ptr_from_list_nodes(next, lane_id);
        count += __popc(__ballot_sync(REGULAR_NODE_KEY_MASK,
                                      src_unit_data != EMPTY_KEY));
        next = __shfl_sync(0xFFFFFFFF, src_unit_data, 31, 32);
    }
    // writing back the results:
    if (lane_id == 0) {
        d_count_result[wid] = count;
    }
}

/*
 * This kernel goes through all allocated bitmaps for a slab_hash_ctx's
 * allocator and store number of allocated slabs.
 * TODO: this should be moved into allocator's codebase (violation of layers)
 */
template <typename KeyT, typename ValueT, typename HashFunc>
__global__ void compute_stats_allocators(
        uint32_t* d_count_super_block,
        SlabHashContext<KeyT, ValueT, HashFunc> slab_hash_ctx) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    int num_bitmaps = slab_hash_ctx.getAllocatorContext()
                              .NUM_MEM_BLOCKS_PER_SUPER_BLOCK_ *
                      32;
    if (tid >= num_bitmaps) {
        return;
    }

    for (int i = 0; i < slab_hash_ctx.getAllocatorContext().num_super_blocks_;
         i++) {
        uint32_t read_bitmap = *(
                slab_hash_ctx.getAllocatorContext().get_ptr_for_bitmap(i, tid));
        atomicAdd(&d_count_super_block[i], __popc(read_bitmap));
    }
}
