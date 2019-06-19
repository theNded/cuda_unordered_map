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

    /******** Hash table meta data ********/
    uint32_t num_elems = 1 << 20;
    float expected_chain = 0.6f;
    uint32_t num_elems_per_bucket = 100;
    uint32_t expected_elems_per_bucket = expected_chain * num_elems_per_bucket;
    uint32_t num_buckets = (num_elems + expected_elems_per_bucket - 1) /
                           expected_elems_per_bucket;

    /******** Insertion data ********/
    using KeyT = uint32_t;
    using ValueT = uint32_t;
    using HashFunc = HasherUint32;
    const auto f = [](const KeyT& key) { return key * 10; };

    std::vector<KeyT> h_key(num_elems);
    std::vector<ValueT> h_value(num_elems);
    const int64_t seed = 1;
    std::mt19937 rng(seed);
    std::vector<uint32_t> index(num_elems);
    std::iota(index.begin(), index.end(), 0);
    std::shuffle(index.begin(), index.end(), rng);
    for (int32_t i = 0; i < index.size(); i++) {
        h_key[i] = index[i];
        h_value[i] = f(h_key[i]);
    }

    /******** Query data (part from first half, part from second half) *******/
    uint32_t num_queries = num_elems / 2;
    float existing_ratio = 0.6f;
    uint32_t num_existing = static_cast<uint32_t>(existing_ratio * num_queries);
    std::vector<KeyT> h_query(num_queries);
    std::vector<ValueT> h_result_gt(num_queries);
    std::vector<ValueT> h_result(num_queries);

    /* from the 1st half */
    for (int i = 0; i < num_existing; i++) {
        h_query[i] = h_key[num_elems / 2 - 1 - i];
        h_result_gt[i] = f(h_query[i]);
    }
    /* from the 2nd half */
    for (int i = 0; i < (num_queries - num_existing); i++) {
        h_query[num_existing + i] = h_key[num_elems / 2 + i];
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


    /******* Instantiate hash table ********/
    int num_insertions = num_elems / 2;
    printf("num elems: %d, num buckets: %d -- num insertions: %d\n", num_elems,
           num_buckets, num_insertions);
    GpuHashTable<KeyT, ValueT, HashFunc> hash_table(num_elems, num_buckets, 0);

    /****** Insert and query first half ********/
    float build_time =
            hash_table.Insert(h_key.data(), h_value.data(), num_elems / 2);
    printf("1) Insert finished in %.3f ms (%.3f M elements/s)\n", build_time,
           double(num_elems / 2) / (build_time * 1000.0));

    /* Expect 0.6 of them found */
    float search_time =
            hash_table.Search(h_query.data(), h_result.data(), num_queries);
    printf("2) Query finished in %.3f ms (%.3f M queries/s)\n",
           search_time, double(num_queries) / (search_time * 1000.0));

    bool search_success = true;
    for (int i = 0; i < num_queries; i++) {
        if (h_result_gt[i] != h_result[i]) {
            printf("### Result at index %d: [%d] -> %d, expected: %d\n",
                   i, h_query[i], h_result[i], h_result_gt[i]);
            search_success = false;
        }
    }
    if (search_success) {
        printf("2) Validation done\n");
    }
    double load_factor = hash_table.measureLoadFactor(1);
    printf("The load factor is %.2f, number of buckets %d\n", load_factor,
           num_buckets);

    /****** Insert and query second half ********/
    build_time = hash_table.Insert(
                    h_key.data() + num_elems / 2,
                    h_value.data() + num_elems / 2, num_elems / 2);
    printf("3) Insert finished in %.3f ms (%.3f M elements/s)\n", build_time,
            double(num_elems / 2) / (build_time * 1000.0));

    /* Expect all of them found */
    search_time =
            hash_table.Search(h_query.data(), h_result.data(), num_queries);
    printf("4) Query finished in %.3f ms (%.3f M queries/s)\n",
           search_time, double(num_queries) / (search_time * 1000.0));
    search_success = true;
    for (int i = 0; i < num_queries; i++) {
        if (f(h_query[i]) != h_result[i]) {
            printf("### Result at index %d: [%d] -> %d, expected: %d\n",
                   i, h_query[i], h_result[i], h_result_gt[i]);
            search_success = false;
        }
    }
    if (search_success) {
        printf("4) Validation done\n");
    }

    load_factor = hash_table.measureLoadFactor(1);
    printf("The load factor is %.2f, number of buckets %d\n", load_factor,
           num_buckets);

    return 0;
}