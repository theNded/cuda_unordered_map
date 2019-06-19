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

#include "../memory_heap/MemoryHeapHost.cuh"

/*
 * This is the main class that will be shallowly copied into the device to be
 * used at runtime. This class does not own the allocated memory on the gpu
 * (i.e., d_table_)
 */
template <typename KeyT, typename ValueT, typename HashFunc>
class GpuSlabHashContext {
public:
    // fixed known parameters:
    static constexpr uint32_t WARP_WIDTH_ = 32;

    GpuSlabHashContext() : num_buckets_(0), d_table_(nullptr) {
        // a single slab on a ConcurrentMap should be 128 bytes
        assert(sizeof(ConcurrentSlab) == (WARP_WIDTH_ * sizeof(uint32_t)));
    }

    static size_t getSlabUnitSize() { return sizeof(ConcurrentSlab); }

    static std::string getSlabHashTypeName() { return "ConcurrentMap"; }

    __host__ void initParameters(const uint32_t num_buckets,
                                 int8_t* d_table,
                                 SlabListAllocatorContext* allocator_ctx,
                                 MemoryHeapContext<KeyT> key_allocator_ctx) {
        num_buckets_ = num_buckets;
        d_table_ = reinterpret_cast<ConcurrentSlab*>(d_table);
        slab_list_allocator_ctx_ = *allocator_ctx;
        key_allocator_ctx_ = key_allocator_ctx;
    }

    __device__ __host__ __forceinline__ SlabListAllocatorContext&
    getAllocatorContext() {
        return slab_list_allocator_ctx_;
    }

    __device__ __host__ __forceinline__ ConcurrentSlab*
    getDeviceTablePointer() {
        return d_table_;
    }

    __device__ __host__ __forceinline__ uint32_t
    computeBucket(const KeyT& key) const {
        return hash_fn_(key) % num_buckets_;
    }

    // threads in a warp cooperate with each other to insert key-value pairs
    // into the slab hash
    __device__ __forceinline__ void insertPair(bool& to_be_inserted,
                                               const uint32_t& lane_id,
                                               const KeyT& myKey,
                                               const ValueT& myValue,
                                               const uint32_t bucket_id);

    // threads in a warp cooeparte with each other to search for keys
    // if found, it returns the corresponding value, else SEARCH_NOT_FOUND
    // is returned
    __device__ __forceinline__ void searchKey(bool& to_be_searched,
                                              const uint32_t& lane_id,
                                              const KeyT& myKey,
                                              ValueT& myValue,
                                              const uint32_t bucket_id);

    // all threads within a warp cooperate with each other to delete keys
    __device__ __forceinline__ void deleteKey(bool& to_be_deleted,
                                              const uint32_t& lane_id,
                                              const KeyT& myKey,
                                              const uint32_t bucket_id);

    __device__ __forceinline__ uint32_t* getPointerFromSlab(
            const SlabAddressT& slab_address, const uint32_t lane_id) {
        return slab_list_allocator_ctx_.getPointerFromSlab(slab_address,
                                                           lane_id);
    }

    __device__ __forceinline__ uint32_t* getPointerFromBucket(
            const uint32_t bucket_id, const uint32_t lane_id) {
        return reinterpret_cast<uint32_t*>(d_table_) +
               bucket_id * BASE_UNIT_SIZE + lane_id;
    }

private:
    // this function should be operated in a warp-wide fashion
    // TODO: add required asserts to make sure this is true in tests/debugs
    __device__ __forceinline__ SlabAllocAddressT
    allocateSlab(const uint32_t& lane_id) {
        return slab_list_allocator_ctx_.warpAllocate(lane_id);
    }

    // a thread-wide function to free the slab that was just allocated
    __device__ __forceinline__ void freeSlab(const SlabAllocAddressT slab_ptr) {
        slab_list_allocator_ctx_.freeUntouched(slab_ptr);
    }

    uint32_t num_buckets_;
    HashFunc hash_fn_;

    ConcurrentSlab* d_table_;
    SlabListAllocatorContext slab_list_allocator_ctx_;
    MemoryHeapContext<KeyT> key_allocator_ctx_;
};

/*
 * This class owns the allocated memory for the hash table
 */
template <typename KeyT, typename ValueT, typename HashFunc>
class GpuSlabHash {
private:
    // fixed known parameters:
    static constexpr uint32_t BLOCKSIZE_ = 128;
    static constexpr uint32_t WARP_WIDTH_ = 32;
    static constexpr uint32_t PRIME_DIVISOR_ = 4294967291u;

    uint32_t num_buckets_;

    // a raw pointer to the initial allocated memory for all buckets
    int8_t* d_table_;
    size_t slab_unit_size_;

    GpuSlabHashContext<KeyT, ValueT, HashFunc> gpu_context_;
    std::shared_ptr<MemoryHeap<KeyT>> key_allocator_;
    std::shared_ptr<SlabListAllocator> slab_list_allocator_;

    uint32_t device_idx_;

public:
    GpuSlabHash(const uint32_t num_buckets,
                const std::shared_ptr<SlabListAllocator>& slab_list_allocator,
                const std::shared_ptr<MemoryHeap<KeyT>> &key_allocator,
                uint32_t device_idx)
        : num_buckets_(num_buckets),
          slab_list_allocator_(slab_list_allocator),
          key_allocator_(key_allocator),
          device_idx_(device_idx),
          d_table_(nullptr),
          slab_unit_size_(0) {
        assert(slab_list_allocator && key_allocator &&
               "No proper dynamic allocator attached to the slab hash.");

        int32_t devCount = 0;
        CHECK_CUDA(cudaGetDeviceCount(&devCount));
        assert(device_idx_ < devCount);

        CHECK_CUDA(cudaSetDevice(device_idx_));

        // initializing the gpu_context_:
        slab_unit_size_ = gpu_context_.getSlabUnitSize();

        // allocating initial buckets:
        CHECK_CUDA(cudaMalloc(&d_table_, slab_unit_size_ * num_buckets_));
        CHECK_CUDA(cudaMemset(d_table_, 0xFF, slab_unit_size_ * num_buckets_));

        gpu_context_.initParameters(num_buckets_, d_table_,
                                    slab_list_allocator_->getContextPtr(),
                                    key_allocator_->gpu_context_);
    }

    ~GpuSlabHash() {
        CHECK_CUDA(cudaSetDevice(device_idx_));
        CHECK_CUDA(cudaFree(d_table_));
    }

    // returns some debug information about the slab hash
    std::string to_string();
    double computeLoadFactor(int flag);

    void buildBulk(KeyT* d_key, ValueT* d_value, uint32_t num_keys);
    void searchIndividual(KeyT* d_query,
                          ValueT* d_result,
                          uint32_t num_queries);
    void searchBulk(KeyT* d_query, ValueT* d_result, uint32_t num_queries);
    void deleteIndividual(KeyT* d_key, uint32_t num_keys);
    void batchedOperation(KeyT* d_key, ValueT* d_result, uint32_t num_ops);
};