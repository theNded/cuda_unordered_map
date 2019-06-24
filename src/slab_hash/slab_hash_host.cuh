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

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
void GpuSlabHash<KeyT, D, ValueT, HashFunc>::buildBulk(KeyTD* d_key,
                                                       ValueT* d_value,
                                                       uint32_t num_keys) {
    const uint32_t num_blocks = (num_keys + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    // calling the kernel for bulk build:
    CHECK_CUDA(cudaSetDevice(device_idx_));
    build_table_kernel<KeyT, D, ValueT, HashFunc><<<num_blocks, BLOCKSIZE_>>>(
            d_key, d_value, num_keys, gpu_context_);
}

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
void GpuSlabHash<KeyT, D, ValueT, HashFunc>::searchIndividual(
        KeyTD* d_query, ValueT* d_result, uint32_t num_queries) {
    CHECK_CUDA(cudaSetDevice(device_idx_));
    const uint32_t num_blocks = (num_queries + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    search_table<KeyT, D, ValueT, HashFunc><<<num_blocks, BLOCKSIZE_>>>(
            d_query, d_result, num_queries, gpu_context_);
}

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
void GpuSlabHash<KeyT, D, ValueT, HashFunc>::deleteIndividual(
        KeyTD* d_key, uint32_t num_keys) {
    CHECK_CUDA(cudaSetDevice(device_idx_));
    const uint32_t num_blocks = (num_keys + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    delete_table_keys<KeyT, D, ValueT, HashFunc>
            <<<num_blocks, BLOCKSIZE_>>>(d_key, num_keys, gpu_context_);
}

// perform a batch of (a mixture of) updates/searches
template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
void GpuSlabHash<KeyT, D, ValueT, HashFunc>::batchedOperation(
        KeyTD* d_key, ValueT* d_result, uint32_t num_ops) {
    CHECK_CUDA(cudaSetDevice(device_idx_));
    const uint32_t num_blocks = (num_ops + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    MixedOperation<KeyT, D, ValueT, HashFunc><<<num_blocks, BLOCKSIZE_>>>(
            d_key, d_result, num_ops, gpu_context_);
}

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
std::string GpuSlabHash<KeyT, D, ValueT, HashFunc>::to_string() {
    std::string result;
    result += " ==== GpuSlabHash: \n";
    result += "\t Running on device \t\t " + std::to_string(device_idx_) + "\n";
    result += "\t SlabHashType:     \t\t " +
              gpu_context_.getSlabHashTypeName() + "\n";
    result +=
            "\t Number of buckets:\t\t " + std::to_string(num_buckets_) + "\n";
    result += "\t d_table_ address: \t\t " +
              std::to_string(reinterpret_cast<uint64_t>(
                      static_cast<void*>(d_table_))) +
              "\n";
    return result;
}

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
double GpuSlabHash<KeyT, D, ValueT, HashFunc>::computeLoadFactor(int flag = 0) {
    uint32_t* h_bucket_count = new uint32_t[num_buckets_];
    uint32_t* d_bucket_count;
    CHECK_CUDA(cudaMalloc((void**)&d_bucket_count,
                          sizeof(uint32_t) * num_buckets_));
    CHECK_CUDA(cudaMemset(d_bucket_count, 0, sizeof(uint32_t) * num_buckets_));

    const auto& dynamic_alloc = gpu_context_.getAllocatorContext();
    const uint32_t num_super_blocks = dynamic_alloc.num_super_blocks_;
    uint32_t* h_count_super_blocks = new uint32_t[num_super_blocks];
    uint32_t* d_count_super_blocks;
    CHECK_CUDA(cudaMalloc((void**)&d_count_super_blocks,
                          sizeof(uint32_t) * num_super_blocks));
    CHECK_CUDA(cudaMemset(d_count_super_blocks, 0,
                          sizeof(uint32_t) * num_super_blocks));
    //---------------------------------
    // counting the number of inserted elements:
    const uint32_t blocksize = 128;
    const uint32_t num_blocks = (num_buckets_ * 32 + blocksize - 1) / blocksize;
    bucket_count_kernel<KeyT, D, ValueT, HashFunc><<<num_blocks, blocksize>>>(
            gpu_context_, d_bucket_count, num_buckets_);
    CHECK_CUDA(cudaMemcpy(h_bucket_count, d_bucket_count,
                          sizeof(uint32_t) * num_buckets_,
                          cudaMemcpyDeviceToHost));

    int total_elements_stored = 0;
    for (int i = 0; i < num_buckets_; i++) {
        total_elements_stored += h_bucket_count[i];
    }

    if (flag) {
        printf("## Total elements stored: %d (%lu bytes).\n",
               total_elements_stored,
               total_elements_stored * (sizeof(KeyT) + sizeof(ValueT)));
    }

    // counting total number of allocated memory units:
    int num_mem_units = dynamic_alloc.NUM_MEM_BLOCKS_PER_SUPER_BLOCK_ * 32;
    int num_cuda_blocks = (num_mem_units + blocksize - 1) / blocksize;
    compute_stats_allocators<<<num_cuda_blocks, blocksize>>>(
            d_count_super_blocks, gpu_context_);

    CHECK_CUDA(cudaMemcpy(h_count_super_blocks, d_count_super_blocks,
                          sizeof(uint32_t) * num_super_blocks,
                          cudaMemcpyDeviceToHost));

    // computing load factor
    int total_mem_units = num_buckets_;
    for (int i = 0; i < num_super_blocks; i++)
        total_mem_units += h_count_super_blocks[i];

    double load_factor =
            double(total_elements_stored * (sizeof(KeyT) + sizeof(ValueT))) /
            double(total_mem_units * WARP_WIDTH * sizeof(uint32_t));

    if (d_count_super_blocks) CHECK_CUDA(cudaFree(d_count_super_blocks));
    if (d_bucket_count) CHECK_CUDA(cudaFree(d_bucket_count));
    delete[] h_bucket_count;
    delete[] h_count_super_blocks;

    return load_factor;
}
