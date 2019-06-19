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

#include "slab_hash/instantiate.cuh"

struct HasherUint32 {
    __device__ __host__ uint32_t operator()(const uint32_t& key) const {
        static const uint32_t hash_x = 1791095845u;
        static const uint32_t hash_y = 4282876139u;
        static const uint32_t prime_devisor = 4294967291u;

        return ((hash_x ^ key) + hash_y) % prime_devisor;
    }
};

/* Lightweight wrapper to handle host input */
template <typename KeyT, typename ValueT, typename HashFunc>
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
    std::shared_ptr<DynamicAllocatorT> dynamic_allocator_;
    std::shared_ptr<GpuSlabHash<KeyT, ValueT, HashFunc>> slab_hash_;

    KeyT* d_key_;
    ValueT* d_value_;
    KeyT* d_query_;
    ValueT* d_result_;

    GpuHashTable(uint32_t max_keys,
                 uint32_t num_buckets,
                 const uint32_t device_idx)
        : max_keys_(max_keys),
          num_buckets_(num_buckets),
          cuda_device_idx_(device_idx),
          slab_hash_(nullptr),
          dynamic_allocator_(nullptr) {
        /* Set device */
        int32_t cuda_device_count_ = 0;
        CHECK_CUDA(cudaGetDeviceCount(&cuda_device_count_));
        assert(cuda_device_idx_ < cuda_device_count_);
        CHECK_CUDA(cudaSetDevice(cuda_device_idx_));

        // allocating key, value arrays:
        CHECK_CUDA(cudaMalloc((void**)&d_key_, sizeof(KeyT) * max_keys_));
        CHECK_CUDA(cudaMalloc((void**)&d_value_, sizeof(ValueT) * max_keys_));
        CHECK_CUDA(cudaMalloc((void**)&d_query_, sizeof(KeyT) * max_keys_));
        CHECK_CUDA(cudaMalloc((void**)&d_result_, sizeof(ValueT) * max_keys_));

        // allocate an initialize the allocator:
        dynamic_allocator_ = std::make_shared<DynamicAllocatorT>();

        // slab hash:
        slab_hash_ = std::make_shared<GpuSlabHash<KeyT, ValueT, HashFunc>>(
                num_buckets_, dynamic_allocator_, cuda_device_idx_);
    }

    ~GpuHashTable() {
        CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
        CHECK_CUDA(cudaFree(d_key_));

        CHECK_CUDA(cudaFree(d_value_));

        CHECK_CUDA(cudaFree(d_query_));
        CHECK_CUDA(cudaFree(d_result_));
    }

    float Insert(KeyT* h_key, ValueT* h_value, uint32_t num_keys) {
        // moving key-values to the device:
        CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
        CHECK_CUDA(cudaMemcpy(d_key_, h_key, sizeof(KeyT) * num_keys,
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

    float Search(KeyT* h_query, ValueT* h_result, uint32_t num_queries) {
        CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
        CHECK_CUDA(cudaMemcpy(d_query_, h_query, sizeof(KeyT) * num_queries,
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

    float Delete(KeyT* h_key, uint32_t num_keys) {
        CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
        CHECK_CUDA(cudaMemcpy(d_key_, h_key, sizeof(KeyT) * num_keys,
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