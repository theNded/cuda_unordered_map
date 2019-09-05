/*
 * Copyright 2018 Saman Ashkiani
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

#include <stdint.h>
#include <iostream>
#include "../helper_cuda.h"
#include "allocator.h"
#include "config.h"
/*
 * This class does not own any memory, and will be shallowly copied into device
 * kernel
 */
class SlabAllocContext {
public:
    static constexpr uint32_t LOG_NUM_MEM_BLOCKS_ = 8;
    static constexpr uint32_t NUM_SUPER_BLOCKS_ALLOCATOR_ = 32;
    static constexpr uint32_t MEM_UNIT_WARP_MULTIPLES_ = 1;

    // fixed parameters for the SlabAlloc
    static constexpr uint32_t NUM_MEM_UNITS_PER_BLOCK_ = 1024;
    static constexpr uint32_t NUM_BITMAP_PER_MEM_BLOCK_ = 32;
    static constexpr uint32_t BITMAP_SIZE_ = 32;
    static constexpr uint32_t WARP_SIZE = 32;
    static constexpr uint32_t MEM_UNIT_SIZE_ =
            MEM_UNIT_WARP_MULTIPLES_ * WARP_SIZE;
    static constexpr uint32_t SUPER_BLOCK_BIT_OFFSET_ALLOC_ = 27;
    static constexpr uint32_t MEM_BLOCK_BIT_OFFSET_ALLOC_ = 10;
    static constexpr uint32_t MEM_UNIT_BIT_OFFSET_ALLOC_ = 5;
    static constexpr uint32_t NUM_MEM_BLOCKS_PER_SUPER_BLOCK_ =
            (1 << LOG_NUM_MEM_BLOCKS_);
    static constexpr uint32_t MEM_BLOCK_SIZE_ =
            NUM_MEM_UNITS_PER_BLOCK_ * MEM_UNIT_SIZE_;
    static constexpr uint32_t SUPER_BLOCK_SIZE_ =
            ((BITMAP_SIZE_ + MEM_BLOCK_SIZE_) *
             NUM_MEM_BLOCKS_PER_SUPER_BLOCK_);
    static constexpr uint32_t MEM_BLOCK_OFFSET_ =
            (BITMAP_SIZE_ * NUM_MEM_BLOCKS_PER_SUPER_BLOCK_);
    static constexpr uint32_t num_super_blocks_ = NUM_SUPER_BLOCKS_ALLOCATOR_;

    SlabAllocContext()
        : super_blocks_(nullptr),
          hash_coef_(0),
          num_attempts_(0),
          resident_index_(0),
          super_block_index_(0),
          allocated_index_(0) {}

    SlabAllocContext& operator=(const SlabAllocContext& rhs) {
        super_blocks_ = rhs.super_blocks_;
        hash_coef_ = rhs.hash_coef_;
        num_attempts_ = 0;
        resident_index_ = 0;
        super_block_index_ = 0;
        allocated_index_ = 0;
        return *this;
    }

    ~SlabAllocContext() {}

    void Setup(uint32_t* super_blocks, uint32_t hash_coef) {
        super_blocks_ = super_blocks;
        hash_coef_ = hash_coef;
    }

    __device__ __forceinline__ uint32_t* get_unit_ptr_from_slab(
            const addr_t& next, const uint32_t& lane_id) {
        return super_blocks_ + addressDecoder(next) + lane_id;
    }
    __device__ __forceinline__ uint32_t* get_ptr_for_bitmap(
            const uint32_t super_block_index, const uint32_t bitmap_index) {
        return super_blocks_ + super_block_index * SUPER_BLOCK_SIZE_ +
               bitmap_index;
    }

    // Objective: each warp selects its own resident warp allocator:
    __device__ void Init(uint32_t& tid, uint32_t& lane_id) {
        // hashing the memory block to be used:
        createMemBlockIndex(tid >> 5);

        // loading the assigned memory block:
        resident_bitmap_ =
                *(super_blocks_ + super_block_index_ * SUPER_BLOCK_SIZE_ +
                  resident_index_ * BITMAP_SIZE_ + lane_id);
        allocated_index_ = 0xFFFFFFFF;
    }

    __device__ uint32_t WarpAllocate(const uint32_t& lane_id) {
        // tries and allocate a new memory units within the resident memory
        // block if it returns 0xFFFFFFFF, then there was not any empty memory
        // unit a new resident block should be chosen, and repeat again
        // allocated result:  5  bits: super_block_index
        //                    17 bits: memory block index
        //                    5  bits: memory unit index (hi-bits of 10bit)
        //                    5  bits: memory unit index (lo-bits of 10bit)
        int empty_lane = -1;
        uint32_t free_lane;
        uint32_t read_bitmap = resident_bitmap_;
        uint32_t allocated_result = 0xFFFFFFFF;
        // works as long as <31 bit are used in the allocated_result
        // in other words, if there are 32 super blocks and at most 64k blocks
        // per super block

        while (allocated_result == 0xFFFFFFFF) {
            empty_lane = __ffs(~resident_bitmap_) - 1;
            free_lane = __ballot_sync(0xFFFFFFFF, empty_lane >= 0);
            if (free_lane == 0) {
                // all bitmaps are full: need to be rehashed again:
                updateMemBlockIndex((threadIdx.x + blockIdx.x * blockDim.x) >>
                                    5);
                read_bitmap = resident_bitmap_;
                continue;
            }
            uint32_t src_lane = __ffs(free_lane) - 1;
            if (src_lane == lane_id) {
                read_bitmap = atomicCAS(
                        super_blocks_ + super_block_index_ * SUPER_BLOCK_SIZE_ +
                                resident_index_ * BITMAP_SIZE_ + lane_id,
                        resident_bitmap_, resident_bitmap_ | (1 << empty_lane));
                if (read_bitmap == resident_bitmap_) {
                    // successful attempt:
                    resident_bitmap_ |= (1 << empty_lane);
                    allocated_result =
                            (super_block_index_
                             << SUPER_BLOCK_BIT_OFFSET_ALLOC_) |
                            (resident_index_ << MEM_BLOCK_BIT_OFFSET_ALLOC_) |
                            (lane_id << MEM_UNIT_BIT_OFFSET_ALLOC_) |
                            empty_lane;
                } else {
                    // Not successful: updating the current bitmap
                    resident_bitmap_ = read_bitmap;
                }
            }
            // asking for the allocated result;
            allocated_result =
                    __shfl_sync(0xFFFFFFFF, allocated_result, src_lane);
        }
        return allocated_result;
    }

    // This function, frees a recently allocated memory unit by a single thread.
    // Since it is untouched, there shouldn't be any worries for the actual
    // memory contents to be reset again.
    __device__ void FreeUntouched(addr_t ptr) {
        atomicAnd(super_blocks_ + getSuperBlockIndex(ptr) * SUPER_BLOCK_SIZE_ +
                          getMemBlockIndex(ptr) * BITMAP_SIZE_ +
                          (getMemUnitIndex(ptr) >> 5),
                  ~(1 << (getMemUnitIndex(ptr) & 0x1F)));
    }

private:
    // =========
    // some helper inline address functions:
    // =========
    __device__ __host__ __forceinline__ uint32_t
    getSuperBlockIndex(addr_t address) const {
        return address >> SUPER_BLOCK_BIT_OFFSET_ALLOC_;
    }
    __device__ __host__ __forceinline__ uint32_t
    getMemBlockIndex(addr_t address) const {
        return ((address >> MEM_BLOCK_BIT_OFFSET_ALLOC_) & 0x1FFFF);
    }
    __device__ __host__ __forceinline__ addr_t
    getMemBlockAddress(addr_t address) const {
        return (MEM_BLOCK_OFFSET_ +
                getMemBlockIndex(address) * MEM_BLOCK_SIZE_);
    }
    __device__ __host__ __forceinline__ uint32_t
    getMemUnitIndex(addr_t address) const {
        return address & 0x3FF;
    }
    __device__ __host__ __forceinline__ addr_t
    getMemUnitAddress(addr_t address) {
        return getMemUnitIndex(address) * MEM_UNIT_SIZE_;
    }

    // called at the beginning of the kernel:
    __device__ void createMemBlockIndex(uint32_t global_warp_id) {
        super_block_index_ = global_warp_id % num_super_blocks_;
        resident_index_ =
                (hash_coef_ * global_warp_id) >> (32 - LOG_NUM_MEM_BLOCKS_);
    }

    // called when the allocator fails to find an empty unit to allocate:
    __device__ void updateMemBlockIndex(uint32_t global_warp_id) {
        num_attempts_++;
        super_block_index_++;
        super_block_index_ = (super_block_index_ == num_super_blocks_)
                                     ? 0
                                     : super_block_index_;
        resident_index_ = (hash_coef_ * (global_warp_id + num_attempts_)) >>
                          (32 - LOG_NUM_MEM_BLOCKS_);
        // loading the assigned memory block:
        resident_bitmap_ =
                *((super_blocks_ + super_block_index_ * SUPER_BLOCK_SIZE_) +
                  resident_index_ * BITMAP_SIZE_ + (threadIdx.x & 0x1f));
    }

    __host__ __device__ addr_t addressDecoder(addr_t address_ptr_index) {
        return getSuperBlockIndex(address_ptr_index) * SUPER_BLOCK_SIZE_ +
               getMemBlockAddress(address_ptr_index) +
               getMemUnitIndex(address_ptr_index) * MEM_UNIT_WARP_MULTIPLES_ *
                       WARP_SIZE;
    }

    __host__ __device__ void print_address(addr_t address_ptr_index) {
        printf("Super block Index: %d, Memory block index: %d, Memory unit "
               "index: "
               "%d\n",
               getSuperBlockIndex(address_ptr_index),
               getMemBlockIndex(address_ptr_index),
               getMemUnitIndex(address_ptr_index));
    }

private:
    // a pointer to each super-block
    uint32_t* super_blocks_;

    // hash_coef (register): used as (16 bits, 16 bits) for hashing
    uint32_t hash_coef_;  // a random 32-bit

    // resident_index: (register)
    // should indicate what memory block and super block is currently resident
    // (16 bits       + 5 bits)
    // (memory block  + super block)
    uint32_t num_attempts_;
    uint32_t resident_index_;
    uint32_t resident_bitmap_;
    uint32_t super_block_index_;
    uint32_t allocated_index_;  // to be asked via shuffle after
};

__global__ void compute_stats_allocators(uint32_t* d_count_super_block,
                                         SlabAllocContext ctx);
__global__ void CountSlabsPerSuperblockKernel(SlabAllocContext context,
                                              uint32_t* slabs_per_superblock);

/*
 * This class owns the memory for the allocator on the device
 */
template <class _Alloc>
class SlabAlloc {
private:
    // a pointer to each super-block
    uint32_t* super_blocks_;

    // hash_coef (register): used as (16 bits, 16 bits) for hashing
    uint32_t hash_coef_;  // a random 32-bit

    // the context class is actually copied shallowly into GPU device
    SlabAllocContext slab_alloc_context_;
    std::shared_ptr<_Alloc> allocator_;

public:
    SlabAlloc() : super_blocks_(nullptr), hash_coef_(0) {
        // random coefficients for allocator's hash function
        std::mt19937 rng(time(0));
        hash_coef_ = rng();

        allocator_ = std::make_shared<_Alloc>();
        // In the light version, we put num_super_blocks super blocks within
        // a single array
        super_blocks_ = allocator_->template allocate<uint32_t>(
                slab_alloc_context_.SUPER_BLOCK_SIZE_ *
                slab_alloc_context_.num_super_blocks_);

        for (int i = 0; i < slab_alloc_context_.num_super_blocks_; i++) {
            // setting bitmaps into zeros:
            CHECK_CUDA(cudaMemset(
                    super_blocks_ + i * slab_alloc_context_.SUPER_BLOCK_SIZE_,
                    0x00,
                    slab_alloc_context_.NUM_MEM_BLOCKS_PER_SUPER_BLOCK_ *
                            slab_alloc_context_.BITMAP_SIZE_ *
                            sizeof(uint32_t)));
            // setting empty memory units into ones:
            CHECK_CUDA(cudaMemset(
                    super_blocks_ + i * slab_alloc_context_.SUPER_BLOCK_SIZE_ +
                            (slab_alloc_context_
                                     .NUM_MEM_BLOCKS_PER_SUPER_BLOCK_ *
                             slab_alloc_context_.BITMAP_SIZE_),
                    0xFF,
                    slab_alloc_context_.MEM_BLOCK_SIZE_ *
                            slab_alloc_context_
                                    .NUM_MEM_BLOCKS_PER_SUPER_BLOCK_ *
                            sizeof(uint32_t)));
        }

        // initializing the slab context:
        slab_alloc_context_.Setup(super_blocks_, hash_coef_);
    }
    ~SlabAlloc() { allocator_->template deallocate<uint32_t>(super_blocks_); }

    SlabAllocContext& getContext() { return slab_alloc_context_; }

    int CountSlabsPerSuperblock() {
        const uint32_t num_super_blocks = slab_alloc_context_.num_super_blocks_;
        uint32_t* h_count_super_blocks = new uint32_t[num_super_blocks];
        uint32_t* d_count_super_blocks =
                allocator_->template allocate<uint32_t>(num_super_blocks);
        CHECK_CUDA(cudaMemset(d_count_super_blocks, 0,
                              sizeof(uint32_t) * num_super_blocks));

        // counting total number of allocated memory units:
        int blocksize = 128;
        int num_mem_units =
                slab_alloc_context_.NUM_MEM_BLOCKS_PER_SUPER_BLOCK_ * 32;
        int num_cuda_blocks = (num_mem_units + blocksize - 1) / blocksize;
        compute_stats_allocators<<<num_cuda_blocks, blocksize>>>(
                d_count_super_blocks, slab_alloc_context_);

        CHECK_CUDA(cudaMemcpy(h_count_super_blocks, d_count_super_blocks,
                              sizeof(uint32_t) * num_super_blocks,
                              cudaMemcpyDeviceToHost));

        // computing load factor
        int total_mem_units = 0;
        for (int i = 0; i < num_super_blocks; i++)
            total_mem_units += h_count_super_blocks[i];

        allocator_->template deallocate<uint32_t>(d_count_super_blocks);
        delete[] h_count_super_blocks;

        return total_mem_units;
    }
};

/*
 * This kernel goes through all allocated bitmaps for a slab_hash_ctx's
 * allocator and store number of allocated slabs.
 * TODO: this should be moved into allocator's codebase (violation of
 * layers)
 */
__global__ void compute_stats_allocators(uint32_t* d_count_super_block,
                                         SlabAllocContext ctx) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    int num_bitmaps = ctx.NUM_MEM_BLOCKS_PER_SUPER_BLOCK_ * 32;
    if (tid >= num_bitmaps) {
        return;
    }

    for (int i = 0; i < ctx.num_super_blocks_; i++) {
        uint32_t read_bitmap = *(ctx.get_ptr_for_bitmap(i, tid));
        atomicAdd(&d_count_super_block[i], __popc(read_bitmap));
    }
}

__global__ void CountSlabsPerSuperblockKernel(SlabAllocContext context,
                                              uint32_t* slabs_per_superblock) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    int num_bitmaps = context.NUM_MEM_BLOCKS_PER_SUPER_BLOCK_ * 32;
    if (tid >= num_bitmaps) {
        return;
    }

    for (int i = 0; i < context.num_super_blocks_; i++) {
        uint32_t read_bitmap = *(context.get_ptr_for_bitmap(i, tid));
        atomicAdd(&slabs_per_superblock[i], __popc(read_bitmap));
    }
}
