/*
 * Copyright 2019 Saman Ashkiani
 * Modified by Wei Dong (2019)
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
 *
 */

#pragma once

#include "memory_heap/memory_heap_host.cuh"
#include "slab_hash/instantiate.cuh"

/* Lightweight wrapper to handle host input */
/* KeyT a elementary types: int, long, etc. */
template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
class GpuHashTable {
private:
    uint32_t max_keys_;
    uint32_t num_buckets_;
    int64_t seed_;
    bool req_values_;
    bool identity_hash_;

public:
    uint32_t cuda_device_idx_;

    /* Allocator for the @slab linked lists */
    typedef Coordinate<KeyT, D> KeyTD;
    std::shared_ptr<MemoryHeap<KeyTD>> key_allocator_;
    std::shared_ptr<MemoryHeap<ValueT>> value_allocator_;
    std::shared_ptr<SlabListAllocator> slab_list_allocator_;
    std::shared_ptr<GpuSlabHash<KeyT, D, ValueT, HashFunc>> slab_hash_;

    KeyTD* d_key_;
    ValueT* d_value_;
    KeyTD* d_query_;
    ValueT* d_result_;

    GpuHashTable(uint32_t max_keys,
                 uint32_t num_buckets,
                 const uint32_t device_idx)
        : max_keys_(max_keys),
          num_buckets_(num_buckets),
          cuda_device_idx_(device_idx),
          slab_hash_(nullptr),
          slab_list_allocator_(nullptr) {
        /* Set device */
        int32_t cuda_device_count_ = 0;
        CHECK_CUDA(cudaGetDeviceCount(&cuda_device_count_));
        assert(cuda_device_idx_ < cuda_device_count_);
        CHECK_CUDA(cudaSetDevice(cuda_device_idx_));

        // allocating key, value arrays:
        CHECK_CUDA(cudaMalloc((void**)&d_key_, sizeof(KeyTD) * max_keys_));
        CHECK_CUDA(cudaMalloc((void**)&d_value_, sizeof(ValueT) * max_keys_));
        CHECK_CUDA(cudaMalloc((void**)&d_query_, sizeof(KeyTD) * max_keys_));
        CHECK_CUDA(cudaMalloc((void**)&d_result_, sizeof(ValueT) * max_keys_));

        // allocate an initialize the allocator:
        slab_list_allocator_ = std::make_shared<SlabListAllocator>();
        key_allocator_ = std::make_shared<MemoryHeap<KeyTD>>(max_keys_);
        value_allocator_ = std::make_shared<MemoryHeap<ValueT>>(max_keys_);

        slab_hash_ = std::make_shared<GpuSlabHash<KeyT, D, ValueT, HashFunc>>(
                num_buckets_, slab_list_allocator_, key_allocator_,
                value_allocator_, cuda_device_idx_);
    }

    ~GpuHashTable() {
        CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
        CHECK_CUDA(cudaFree(d_key_));

        CHECK_CUDA(cudaFree(d_value_));

        CHECK_CUDA(cudaFree(d_query_));
        CHECK_CUDA(cudaFree(d_result_));
    }

    float Insert(KeyTD* h_key, ValueT* h_value, uint32_t num_keys) {
        // moving key-values to the device:
        CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
        CHECK_CUDA(cudaMemcpy(d_key_, h_key, sizeof(KeyTD) * num_keys,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_value_, h_value, sizeof(ValueT) * num_keys,
                              cudaMemcpyHostToDevice));

        float temp_time = 0.0f;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);

        slab_hash_->buildBulk(d_key_, d_value_, num_keys);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&temp_time, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return temp_time;
    }

    float Search(KeyTD* h_query, ValueT* h_result, uint32_t num_queries) {
        CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
        CHECK_CUDA(cudaMemcpy(d_query_, h_query, sizeof(KeyTD) * num_queries,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_result_, 0xFF, sizeof(ValueT) * num_queries));

        float temp_time = 0.0f;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        slab_hash_->searchIndividual(d_query_, d_result_, num_queries);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&temp_time, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        CHECK_CUDA(cudaMemcpy(h_result, d_result_, sizeof(ValueT) * num_queries,
                              cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        return temp_time;
    }

    float Delete(KeyTD* h_key, uint32_t num_keys) {
        CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
        CHECK_CUDA(cudaMemcpy(d_key_, h_key, sizeof(KeyTD) * num_keys,
                              cudaMemcpyHostToDevice));

        float temp_time = 0.0f;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        slab_hash_->deleteIndividual(d_key_, num_keys);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&temp_time, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return temp_time;
    }

    float MixedOperation(uint32_t* h_batch_op,
                         uint32_t* h_results,
                         uint32_t batch_size,
                         uint32_t batch_id) {
        CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
        CHECK_CUDA(cudaMemcpy(d_key_ + batch_id * batch_size, h_batch_op,
                              sizeof(uint32_t) * batch_size,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_result_ + batch_id * batch_size, 0xFF,
                              sizeof(uint32_t) * batch_size));

        float temp_time = 0.0f;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        slab_hash_->batchedOperation(d_key_ + batch_id * batch_size, d_result_,
                                     batch_size);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&temp_time, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        CHECK_CUDA(cudaMemcpy(h_results + batch_id * batch_size,
                              d_result_ + batch_id * batch_size,
                              sizeof(uint32_t) * batch_size,
                              cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        return temp_time;
    }

    float measureLoadFactor(int flag = 0) {
        return slab_hash_->computeLoadFactor(flag);
    }
};
