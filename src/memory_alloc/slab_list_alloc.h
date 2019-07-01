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
#include "config.h"
#include "helper_cuda.h"

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
        : d_super_blocks_(nullptr),
          hash_coef_(0),
          num_attempts_(0),
          resident_index_(0),
          super_block_index_(0),
          allocated_index_(0) {}

    SlabAllocContext& operator=(const SlabAllocContext& rhs) {
        d_super_blocks_ = rhs.d_super_blocks_;
        hash_coef_ = rhs.hash_coef_;
        num_attempts_ = 0;
        resident_index_ = 0;
        super_block_index_ = 0;
        allocated_index_ = 0;
        return *this;
    }

    ~SlabAllocContext() {}

    void Init(uint32_t* d_super_block, uint32_t hash_coef) {
        d_super_blocks_ = d_super_block;
        hash_coef_ = hash_coef;
    }

    // =========
    // some helper inline address functions:
    // =========
    __device__ __host__ uint32_t
    getSuperBlockIndex(addr_t address) const;

    __device__ __host__ uint32_t
    getMemBlockIndex(addr_t address) const;

    __device__ __host__ addr_t
    getMemBlockAddress(addr_t address) const;

    __device__ __host__ uint32_t
    getMemUnitIndex(addr_t address) const;

    __device__ __host__ addr_t
    getMemUnitAddress(addr_t address);

    __device__ uint32_t* getPointerFromSlab(const addr_t& next,
                                            const uint32_t& lane_id);

    __device__ uint32_t* getPointerForBitmap(const uint32_t super_block_index,
                                             const uint32_t bitmap_index);

    // called at the beginning of the kernel:
    __device__ void createMemBlockIndex(uint32_t global_warp_id);

    // called when the allocator fails to find an empty unit to allocate:
    __device__ void updateMemBlockIndex(uint32_t global_warp_id);

    // Objective: each warp selects its own resident warp allocator:
    __device__ void initAllocator(uint32_t& tid, uint32_t& lane_id);

    __device__ uint32_t warpAllocate(const uint32_t& lane_id);
    __device__ uint32_t warpAllocateBulk(uint32_t& lane_id, const uint32_t k);

    /*
    This function, frees a recently allocated memory unit by a single thread.
    Since it is untouched, there shouldn't be any worries for the actual memory
    contents to be reset again.
  */
    __device__ void freeUntouched(addr_t ptr);

    __host__ __device__ addr_t
    addressDecoder(addr_t address_ptr_index);

    __host__ __device__ void print_address(addr_t address_ptr_index);

private:
    // a pointer to each super-block
    uint32_t* d_super_blocks_;

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

/*
 * This class owns the memory for the allocator on the device
 */
class SlabAlloc {
private:
    // a pointer to each super-block
    uint32_t* d_super_blocks_;

    // hash_coef (register): used as (16 bits, 16 bits) for hashing
    uint32_t hash_coef_;  // a random 32-bit

    // the context class is actually copied shallowly into GPU device
    SlabAllocContext slab_alloc_context_;

public:
    // =========
    // constructor:
    // =========
    SlabAlloc() : d_super_blocks_(nullptr), hash_coef_(0) {
        // random coefficients for allocator's hash function
        std::mt19937 rng(time(0));
        hash_coef_ = rng();

        // In the light version, we put num_super_blocks super blocks within a
        // single array
        CHECK_CUDA(cudaMalloc((void**)&d_super_blocks_,
                              slab_alloc_context_.SUPER_BLOCK_SIZE_ *
                                      slab_alloc_context_.num_super_blocks_ *
                                      sizeof(uint32_t)));

        for (int i = 0; i < slab_alloc_context_.num_super_blocks_; i++) {
            // setting bitmaps into zeros:
            CHECK_CUDA(cudaMemset(
                    d_super_blocks_ + i * slab_alloc_context_.SUPER_BLOCK_SIZE_,
                    0x00,
                    slab_alloc_context_.NUM_MEM_BLOCKS_PER_SUPER_BLOCK_ *
                            slab_alloc_context_.BITMAP_SIZE_ *
                            sizeof(uint32_t)));
            // setting empty memory units into ones:
            CHECK_CUDA(cudaMemset(
                    d_super_blocks_ +
                            i * slab_alloc_context_.SUPER_BLOCK_SIZE_ +
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
        slab_alloc_context_.Init(d_super_blocks_, hash_coef_);
    }

    // =========
    // destructor:
    // =========
    ~SlabAlloc() { CHECK_CUDA(cudaFree(d_super_blocks_)); }

    // =========
    // Helper member functions:
    // =========
    SlabAllocContext* getContextPtr() { return &slab_alloc_context_; }
};

/** Internal addresses managed by memory_alloc **/
struct KeyValueAddrPair {
    uint32_t key;
    uint32_t value;
};

struct ConcurrentSlab {
    // 15 x 2 + 2
    KeyValueAddrPair data[15];
    uint32_t ptr_index[2];
};
