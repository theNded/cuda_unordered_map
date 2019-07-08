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

#include "coordinate_hash_map.cuh"
#include "coordinate_indexer.h"

template <size_t D>
CoordinateIndexer<D>::CoordinateIndexer(uint32_t max_keys,
                                        uint32_t keys_per_bucket,
                                        float expected_occupancy_per_bucket,
                                        const uint32_t device_idx) {
    hash_map_ = std::make_shared<CoordinateHashMap<KeyT, D, ValueT, HashFunc>>(
            max_keys, keys_per_bucket, expected_occupancy_per_bucket,
            device_idx);
}

template <size_t D>
CoordinateIndexer<D>::~CoordinateIndexer() {}

__global__ void IOTAKernel(uint32_t* values,
                           uint32_t num_keys,
                           uint32_t start_value) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < num_keys) {
        values[tid] = tid + start_value;
    }
}

template <size_t D>
float CoordinateIndexer<D>::Build(KeyT* keys, uint32_t num_keys) {
    /* Prepare indices */
    ValueT* values;
    CHECK_CUDA(cudaMalloc(&values, sizeof(ValueT) * num_keys));
    int threads = 128;
    int blocks = (num_keys + threads - 1) / threads;
    IOTAKernel<<<blocks, threads>>>(values, num_keys, 0);
    CHECK_CUDA(cudaDeviceSynchronize());

    float time = hash_map_->Insert(keys, values, num_keys);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(values));

    return time;
}

template <size_t D>
float CoordinateIndexer<D>::Search(KeyT* query_keys,
                                   ValueT* query_values,
                                   uint8_t* mask,
                                   int num_keys) {
    return hash_map_->Search(query_keys, query_values, mask, num_keys);
}

template <size_t D>
float CoordinateIndexer<D>::ComputeLoadFactor(int flag /* = 0 */) {
    return hash_map_->ComputeLoadFactor(flag);
}
