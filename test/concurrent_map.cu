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

#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>
#include "gpu_hash_table.cuh"

//=======================================
#define DEVICE_ID 0

int main(int argc, char** argv) {
    //=========
    int devCount;
    cudaGetDeviceCount(&devCount);
    cudaDeviceProp devProp;
    if (devCount) {
        cudaSetDevice(DEVICE_ID);  // be changed later
        cudaGetDeviceProperties(&devProp, DEVICE_ID);
    }
    printf("Device: %s\n", devProp.name);

    /** Hash table meta data **/
    uint32_t num_keys = 1 << 20;

    float expected_chain = 0.6f;
    uint32_t num_elems_per_bucket = 100;
    uint32_t expected_elems_per_bucket = expected_chain * num_elems_per_bucket;
    uint32_t num_buckets = (num_keys + expected_elems_per_bucket - 1) /
                           expected_elems_per_bucket;

    /** Generate key-value pairs for building **/
    float existing_ratio = 0.6f;  // ratio of queries within the table
    uint32_t num_queries = num_keys;
    auto num_elements = 2 * num_keys;

    using KeyT = uint32_t;
    using ValueT = uint32_t;
    const auto f = [](const KeyT& key) { return key * 10; };

    std::vector<KeyT> h_key(num_elements);
    std::vector<ValueT> h_value(num_elements);
    std::vector<KeyT> h_query(num_queries);
    std::vector<ValueT> h_result_gt(num_queries);
    std::vector<ValueT> h_result(num_queries);

    const int64_t seed = 1;
    std::mt19937 rng(seed);
    std::vector<uint32_t> index(num_elements);
    std::iota(index.begin(), index.end(), 0);
    std::shuffle(index.begin(), index.end(), rng);

    for (int32_t i = 0; i < index.size(); i++) {
        h_key[i] = index[i];
        h_value[i] = f(h_key[i]);
    }

    /** Generate key-value pairs for query **/
    uint32_t num_existing = static_cast<uint32_t>(existing_ratio * num_queries);

    for (int i = 0; i < num_existing; i++) {
        h_query[i] = h_key[num_keys - 1 - i];
        h_result_gt[i] = f(h_query[i]);
    }

    for (int i = 0; i < (num_queries - num_existing); i++) {
        h_query[num_existing + i] = h_key[num_keys + i];
        h_result_gt[num_existing + i] = SEARCH_NOT_FOUND;
    }

    /* shuffle */
    std::vector<int> q_index(num_queries);
    std::iota(q_index.begin(), q_index.end(), 0);
    std::shuffle(q_index.begin(), q_index.end(), rng);
    for (int i = 0; i < num_queries; i++) {
        std::swap(h_query[i], h_query[q_index[i]]);
        std::swap(h_result_gt[i], h_result_gt[q_index[i]]);
    }

    /** Instantiate hash table **/
    GpuHashTable<KeyT, ValueT> hash_table(num_keys, num_buckets, 0, seed);

    float build_time =
            hash_table.Insert(h_key.data(), h_value.data(), num_keys);
    float search_time = hash_table.Search(h_query.data(), h_result.data(),
                                               num_queries);
    // ==== validation:
    for (int i = 0; i < num_queries; i++) {
        if (h_result_gt[i] != h_result[i]) {
            printf("### wrong result at index %d: [%d] -> %d, but should be "
                   "%d\n",
                   i, h_query[i], h_result[i], h_result_gt[i]);
        }
        if (i == (num_queries - 1)) printf("Validation done successfully\n");
    }

    printf("Hash table: \n");
    printf("num_keys = %d, num_buckets = %d\n", num_keys, num_buckets);
    printf("\t2) Hash table built in %.3f ms (%.3f M elements/s)\n", build_time,
           double(num_keys) / build_time / 1000.0);
    printf("\t3) Hash table search (%.2f) in %.3f ms (%.3f M queries/s)\n",
           existing_ratio, search_time,
           double(num_queries) / search_time / 1000.0);
    double load_factor = hash_table.measureLoadFactor();

    printf("The load factor is %.2f, number of buckets %d\n", load_factor,
           num_buckets);

    return 0;
}