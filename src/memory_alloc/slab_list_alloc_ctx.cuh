//
// Created by dongw1 on 7/1/19.
//

#pragma once

#include "slab_list_alloc.h"

// =========
// some helper inline address functions:
// =========
__device__ __host__ uint32_t
SlabAllocContext::getSuperBlockIndex(addr_t address) const {
    return address >> SUPER_BLOCK_BIT_OFFSET_ALLOC_;
}

__device__ __host__ uint32_t
SlabAllocContext::getMemBlockIndex(addr_t address) const {
    return ((address >> MEM_BLOCK_BIT_OFFSET_ALLOC_) & 0x1FFFF);
}

__device__ __host__ addr_t
SlabAllocContext::getMemBlockAddress(addr_t address) const {
    return (MEM_BLOCK_OFFSET_ + getMemBlockIndex(address) * MEM_BLOCK_SIZE_);
}

__device__ __host__ uint32_t
SlabAllocContext::getMemUnitIndex(addr_t address) const {
    return address & 0x3FF;
}

__device__ __host__ addr_t
SlabAllocContext::getMemUnitAddress(addr_t address) {
    return getMemUnitIndex(address) * MEM_UNIT_SIZE_;
}

__device__ uint32_t* SlabAllocContext::getPointerFromSlab(
        const addr_t& next, const uint32_t& lane_id) {
    return reinterpret_cast<uint32_t*>(d_super_blocks_) + addressDecoder(next) +
           lane_id;
}

__device__ uint32_t* SlabAllocContext::getPointerForBitmap(
        const uint32_t super_block_index, const uint32_t bitmap_index) {
    return d_super_blocks_ + super_block_index * SUPER_BLOCK_SIZE_ +
           bitmap_index;
}

// called at the beginning of the kernel:
__device__ void SlabAllocContext::createMemBlockIndex(uint32_t global_warp_id) {
    super_block_index_ = global_warp_id % num_super_blocks_;
    resident_index_ =
            (hash_coef_ * global_warp_id) >> (32 - LOG_NUM_MEM_BLOCKS_);
}

// called when the allocator fails to find an empty unit to allocate:
__device__ void SlabAllocContext::updateMemBlockIndex(uint32_t global_warp_id) {
    num_attempts_++;
    super_block_index_++;
    super_block_index_ =
            (super_block_index_ == num_super_blocks_) ? 0 : super_block_index_;
    resident_index_ = (hash_coef_ * (global_warp_id + num_attempts_)) >>
                      (32 - LOG_NUM_MEM_BLOCKS_);
    // loading the assigned memory block:
    resident_bitmap_ =
            *((d_super_blocks_ + super_block_index_ * SUPER_BLOCK_SIZE_) +
              resident_index_ * BITMAP_SIZE_ + (threadIdx.x & 0x1f));
}

// Objective: each warp selects its own resident warp allocator:
__device__ void SlabAllocContext::initAllocator(uint32_t& tid,
                                                uint32_t& lane_id) {
    // hashing the memory block to be used:
    createMemBlockIndex(tid >> 5);

    // loading the assigned memory block:
    resident_bitmap_ =
            *(d_super_blocks_ + super_block_index_ * SUPER_BLOCK_SIZE_ +
              resident_index_ * BITMAP_SIZE_ + lane_id);
    allocated_index_ = 0xFFFFFFFF;
}

__device__ uint32_t SlabAllocContext::warpAllocate(const uint32_t& lane_id) {
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
            updateMemBlockIndex((threadIdx.x + blockIdx.x * blockDim.x) >> 5);
            read_bitmap = resident_bitmap_;
            continue;
        }
        uint32_t src_lane = __ffs(free_lane) - 1;
        if (src_lane == lane_id) {
            read_bitmap = atomicCAS(
                    d_super_blocks_ + super_block_index_ * SUPER_BLOCK_SIZE_ +
                            resident_index_ * BITMAP_SIZE_ + lane_id,
                    resident_bitmap_, resident_bitmap_ | (1 << empty_lane));
            if (read_bitmap == resident_bitmap_) {
                // successful attempt:
                resident_bitmap_ |= (1 << empty_lane);
                allocated_result =
                        (super_block_index_ << SUPER_BLOCK_BIT_OFFSET_ALLOC_) |
                        (resident_index_ << MEM_BLOCK_BIT_OFFSET_ALLOC_) |
                        (lane_id << MEM_UNIT_BIT_OFFSET_ALLOC_) | empty_lane;
            } else {
                // Not successful: updating the current bitmap
                resident_bitmap_ = read_bitmap;
            }
        }
        // asking for the allocated result;
        allocated_result = __shfl_sync(0xFFFFFFFF, allocated_result, src_lane);
    }
    return allocated_result;
}

__device__ uint32_t SlabAllocContext::warpAllocateBulk(uint32_t& lane_id,
                                                       const uint32_t k) {
    // tries and allocate k consecutive memory units within the resident
    // memory block if it returns 0xFFFFFFFF, then there was not any empty
    // memory unit a new resident block should be chosen, and repeat again
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
        empty_lane =
                32 -
                (__ffs(__brev(~resident_bitmap_)));  // reversing the order of
        // assigning lanes compared
        // to single allocations
        const uint32_t mask = ((1 << k) - 1) << (empty_lane - k + 1);
        // mask = %x\n", context.resident_bitmap, empty_lane, mask);
        free_lane = __ballot_sync(
                0xFFFFFFFF,
                (empty_lane >= (k - 1)) &&
                        !(resident_bitmap_ &
                          mask));  // update true statement to make sure
        // everything fits
        if (free_lane == 0) {
            // all bitmaps are full: need to be rehashed again:
            updateMemBlockIndex((threadIdx.x + blockIdx.x * blockDim.x) >> 5);
            read_bitmap = resident_bitmap_;
            continue;
        }
        uint32_t src_lane = __ffs(free_lane) - 1;

        if (src_lane == lane_id) {
            read_bitmap = atomicCAS(
                    d_super_blocks_ + super_block_index_ * SUPER_BLOCK_SIZE_ +
                            resident_index_ * BITMAP_SIZE_ + lane_id,
                    resident_bitmap_, resident_bitmap_ | mask);
            if (read_bitmap == resident_bitmap_) {
                // successful attempt:
                resident_bitmap_ |= mask;
                allocated_result =
                        (super_block_index_ << SUPER_BLOCK_BIT_OFFSET_ALLOC_) |
                        (resident_index_ << MEM_BLOCK_BIT_OFFSET_ALLOC_) |
                        (lane_id << MEM_UNIT_BIT_OFFSET_ALLOC_) | empty_lane;
            } else {
                // Not successful: updating the current bitmap
                resident_bitmap_ = read_bitmap;
            }
        }
        // asking for the allocated result;
        allocated_result = __shfl_sync(0xFFFFFFFF, allocated_result, src_lane);
    }
    return allocated_result;
}

/*
This function, frees a recently allocated memory unit by a single thread.
Since it is untouched, there shouldn't be any worries for the actual memory
contents to be reset again.
*/
__device__ void SlabAllocContext::freeUntouched(addr_t ptr) {
    atomicAnd(d_super_blocks_ + getSuperBlockIndex(ptr) * SUPER_BLOCK_SIZE_ +
                      getMemBlockIndex(ptr) * BITMAP_SIZE_ +
                      (getMemUnitIndex(ptr) >> 5),
              ~(1 << (getMemUnitIndex(ptr) & 0x1F)));
}

__host__ __device__ addr_t
SlabAllocContext::addressDecoder(addr_t address_ptr_index) {
    return getSuperBlockIndex(address_ptr_index) * SUPER_BLOCK_SIZE_ +
           getMemBlockAddress(address_ptr_index) +
           getMemUnitIndex(address_ptr_index) * MEM_UNIT_WARP_MULTIPLES_ *
                   WARP_SIZE;
}

__host__ __device__ void SlabAllocContext::print_address(
        addr_t address_ptr_index) {
    printf("Super block Index: %d, Memory block index: %d, Memory unit "
           "index: "
           "%d\n",
           getSuperBlockIndex(address_ptr_index),
           getMemBlockIndex(address_ptr_index),
           getMemUnitIndex(address_ptr_index));
}