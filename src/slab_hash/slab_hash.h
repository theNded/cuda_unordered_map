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

#include "../memory_heap/memory_heap_host.cuh"

template <typename T, size_t D>
struct Coordinate {
private:
    T data_[D];

public:
    __device__ __host__ T& operator[](size_t i) { return data_[i]; }
    __device__ __host__ const T& operator[](size_t i) const { return data_[i]; }

    __device__ __host__ bool operator==(const Coordinate<T, D>& rhs) const {
        bool equal = true;
#pragma unroll 1
        for (size_t i = 0; i < D; ++i) {
            equal &= (data_[i] == rhs[i]);
        }
        return equal;
    }
};

template <typename T, size_t D>
struct CoordinateHashFunc {
    __device__ __host__ uint64_t operator()(const Coordinate<T, D>& key) const {
        uint64_t hash = UINT64_C(14695981039346656037);

        /** We only support 4-byte and 8-byte types **/
        using input_t = typename std::conditional<sizeof(T) == sizeof(uint32_t),
                                                  uint32_t, uint64_t>::type;
#pragma unroll 1
        for (size_t i = 0; i < D; ++i) {
            hash ^= *((input_t*)(&key[i]));
            hash *= UINT64_C(1099511628211);
        }
        return hash;
    }
};

/*
 * This is the main class that will be shallowly copied into the device to be
 * used at runtime. This class does not own the allocated memory on the gpu
 * (i.e., d_table_)
 */
template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
class GpuSlabHashContext {
public:
    typedef Coordinate<KeyT, D> KeyTD;

    // fixed known parameters:
    GpuSlabHashContext() : num_buckets_(0), d_table_(nullptr) {
        // a single slab on a ConcurrentMap should be 128 bytes
        assert(sizeof(ConcurrentSlab) == (WARP_WIDTH * sizeof(uint32_t)));
    }

    static size_t getSlabUnitSize() { return sizeof(ConcurrentSlab); }

    static std::string getSlabHashTypeName() { return "ConcurrentMap"; }

    __host__ void initParameters(
            const uint32_t num_buckets,
            int8_t* d_table,
            SlabListAllocatorContext* allocator_ctx,
            MemoryHeapContext<KeyTD> key_allocator_ctx,
            MemoryHeapContext<ValueT> value_allocator_ctx) {
        num_buckets_ = num_buckets;
        d_table_ = reinterpret_cast<ConcurrentSlab*>(d_table);
        slab_list_allocator_ctx_ = *allocator_ctx;
        key_allocator_ctx_ = key_allocator_ctx;
        value_allocator_ctx_ = value_allocator_ctx;
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
    computeBucket(const KeyTD& key) const {
        return hash_fn_(key) % num_buckets_;
    }

    __device__ void insertPair(bool& to_be_inserted,
                               const uint32_t& lane_id,
                               const KeyTD& myKey,
                               const ValueT& myValue,
                               const uint32_t bucket_id);
    __device__ void searchKey(bool& to_be_searched,
                              const uint32_t& lane_id,
                              const KeyTD& myKey,
                              ValueT& myValue,
                              const uint32_t bucket_id);
    __device__ void deleteKey(bool& to_be_deleted,
                              const uint32_t& lane_id,
                              const KeyTD& myKey,
                              const uint32_t bucket_id);

    __device__ int32_t laneFoundKeyInWarp(const KeyTD& src_key,
                                          uint32_t lane_id,
                                          uint32_t unit_data) {
        bool is_lane_found =
                /* select key lanes */
                ((1 << lane_id) & REGULAR_NODE_KEY_MASK)
                /* validate key addrs */
                && (unit_data != EMPTY_KEY)
                /* find keys in memory heap */
                && key_allocator_ctx_.value_at(unit_data) == src_key;

        return __ffs(__ballot_sync(REGULAR_NODE_KEY_MASK, is_lane_found)) - 1;
    }

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
    MemoryHeapContext<KeyTD> key_allocator_ctx_;
    MemoryHeapContext<ValueT> value_allocator_ctx_;
};

/*
 * This class owns the allocated memory for the hash table
 */
template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
class GpuSlabHash {
public:
    typedef Coordinate<KeyT, D> KeyTD;

private:
    // fixed known parameters:
    static constexpr uint32_t BLOCKSIZE_ = 128;

    uint32_t num_buckets_;

    // a raw pointer to the initial allocated memory for all buckets
    int8_t* d_table_;
    size_t slab_unit_size_;

    GpuSlabHashContext<KeyT, D, ValueT, HashFunc> gpu_context_;
    std::shared_ptr<MemoryHeap<KeyTD>> key_allocator_;
    std::shared_ptr<MemoryHeap<ValueT>> value_allocator_;
    std::shared_ptr<SlabListAllocator> slab_list_allocator_;

    uint32_t device_idx_;

public:
    GpuSlabHash(const uint32_t num_buckets,
                const std::shared_ptr<SlabListAllocator>& slab_list_allocator,
                const std::shared_ptr<MemoryHeap<KeyTD>>& key_allocator,
                const std::shared_ptr<MemoryHeap<ValueT>>& value_allocator,
                uint32_t device_idx)
        : num_buckets_(num_buckets),
          slab_list_allocator_(slab_list_allocator),
          key_allocator_(key_allocator),
          value_allocator_(value_allocator),
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

        gpu_context_.initParameters(
                num_buckets_, d_table_, slab_list_allocator_->getContextPtr(),
                key_allocator_->gpu_context_, value_allocator_->gpu_context_);
    }

    ~GpuSlabHash() {
        CHECK_CUDA(cudaSetDevice(device_idx_));
        CHECK_CUDA(cudaFree(d_table_));
    }

    // returns some debug information about the slab hash
    std::string to_string();
    double computeLoadFactor(int flag);

    void buildBulk(KeyTD* d_key, ValueT* d_value, uint32_t num_keys);
    void searchIndividual(KeyTD* d_query,
                          ValueT* d_result,
                          uint32_t num_queries);
    void searchBulk(KeyTD* d_query, ValueT* d_result, uint32_t num_queries);
    void deleteIndividual(KeyTD* d_key, uint32_t num_keys);
    void batchedOperation(KeyTD* d_key, ValueT* d_result, uint32_t num_ops);
};
