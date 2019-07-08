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
#include "coordinate_hash_map.cuh"

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
    uint32_t num_elems_per_bucket = 15;
    uint32_t expected_elems_per_bucket = expected_chain * num_elems_per_bucket;
    uint32_t num_buckets = (num_elems + expected_elems_per_bucket - 1) /
                           expected_elems_per_bucket;

    /******** Insertion data ********/
    using KeyT = uint32_t;
    constexpr size_t D = 7;
    using KeyTD = Coordinate<KeyT, D>;
    using ValueT = uint32_t;
    using HashFunc = CoordinateHashFunc<KeyT, D>;

    std::vector<KeyTD> h_key(num_elems);
    std::vector<ValueT> h_value(num_elems);
    const int64_t seed = 1;
    std::mt19937 rng(seed);

    std::vector<uint32_t> index(num_elems * D);
    std::iota(index.begin(), index.end(), 0);
    std::shuffle(index.begin(), index.end(), rng);

    for (int32_t i = 0; i < num_elems; i++) {
        for (int d = 0; d < D; ++d) {
            h_key[i][d] = index[i * D + d];
        }
        h_value[i] = i;
    }

    /******** Query data (part from first half, part from second half) *******/
    uint32_t num_queries = num_elems / 2;
    float existing_ratio = 0.6f;
    uint32_t num_existing = static_cast<uint32_t>(existing_ratio * num_queries);
    std::vector<KeyTD> h_query(num_queries);
    std::vector<ValueT> h_result_gt(num_queries);
    std::vector<ValueT> h_result(num_queries);

    std::vector<uint8_t> h_found_gt(num_queries);
    std::vector<uint8_t> h_found(num_queries);

    /* from the 1st half */
    for (int i = 0; i < num_existing; i++) {
        h_query[i] = h_key[num_elems / 2 - 1 - i];
        h_result_gt[i] = h_value[num_elems / 2 - 1 - i];
        h_found_gt[i] = 1;
    }
    /* from the 2nd half */
    for (int i = 0; i < (num_queries - num_existing); i++) {
        h_query[num_existing + i] = h_key[num_elems / 2 + i];
        h_found_gt[num_existing + i] = 0;
    }
    /* shuffle */
    std::vector<int> q_index(num_queries);
    std::iota(q_index.begin(), q_index.end(), 0);
    std::shuffle(q_index.begin(), q_index.end(), rng);
    for (int i = 0; i < num_queries; i++) {
        std::swap(h_query[i], h_query[q_index[i]]);
        std::swap(h_result_gt[i], h_result_gt[q_index[i]]);
        std::swap(h_found_gt[i], h_found_gt[q_index[i]]);
    }

    /******* Instantiate hash table ********/
    int num_insertions = num_elems / 2;
    printf("num elems: %d, num buckets: %d -- num insertions: %d\n", num_elems,
           num_buckets, num_insertions);
    CoordinateHashMap<KeyT, D, ValueT, HashFunc> hash_table(num_elems);

    /****** Insert and query first half ********/
    std::vector<KeyTD> key_1st_half(h_key.begin(), h_key.begin() + num_elems / 2);
    std::vector<ValueT> value_1st_half(h_value.begin(), h_value.begin() + num_elems / 2);
    std::vector<KeyTD> key_2nd_half(h_key.begin() + num_elems / 2, h_key.end());
    std::vector<ValueT> value_2nd_half(h_value.begin() + num_elems / 2, h_value.end());

    float build_time = 0;
    hash_table.Insert(key_1st_half, value_1st_half, build_time);
    printf("1) Insert finished in %.3f ms (%.3f M elements/s)\n", build_time,
           double(num_elems / 2) / (build_time * 1000.0));

    /* Expect 0.6 of them found */
    float search_time = 0;
    hash_table.Search(h_query, h_result, h_found, search_time);
    printf("2) Query finished in %.3f ms (%.3f M queries/s)\n", search_time,
           double(num_queries) / (search_time * 1000.0));

    bool search_success = true;
    for (int i = 0; i < num_queries; i++) {
        if (!h_found_gt[i] && h_found[i]) {
            printf("### wrong result at index %d: should be NOT FOUND\n", i);
            search_success = false;
        }
        if (h_found_gt[i] && !h_found[i]) {
            printf("### wrong result at index %d: should be FOUND\n", i);
            search_success = false;
        }
        if (h_found_gt[i] && h_found[i] && (h_result_gt[i] != h_result[i])) {
            printf("### wrong result at index %d: [%d] -> %d, but should be "
                   "%d\n",
                   i, h_query[i][0], h_result[i], h_result_gt[i]);
            search_success = false;
        }
    }
    if (search_success) {
        printf("2) Validation done\n");
    }
    double load_factor = hash_table.ComputeLoadFactor(1);
    printf("The load factor is %.2f, number of buckets %d\n", load_factor,
           num_buckets);

    /****** Insert and query second half ********/
    hash_table.Insert(key_2nd_half, value_2nd_half, build_time);
    printf("3) Insert finished in %.3f ms (%.3f M elements/s)\n", build_time,
           double(num_elems / 2) / (build_time * 1000.0));

    /* from the 1st half */
    for (int i = 0; i < num_existing; i++) {
        h_query[i] = h_key[num_elems / 2 - 1 - i];
        h_result_gt[i] = h_value[num_elems / 2 - 1 - i];
        h_found_gt[i] = 1;
    }
    /* from the 2nd half */
    for (int i = 0; i < (num_queries - num_existing); i++) {
        h_query[num_existing + i] = h_key[num_elems / 2 + i];
        h_result_gt[num_existing + i] = h_value[num_elems/2 + i];
        h_found_gt[num_existing + i] = 1;
    }
    /* shuffle */
    std::iota(q_index.begin(), q_index.end(), 0);
    std::shuffle(q_index.begin(), q_index.end(), rng);
    for (int i = 0; i < num_queries; i++) {
        std::swap(h_query[i], h_query[q_index[i]]);
        std::swap(h_result_gt[i], h_result_gt[q_index[i]]);
        std::swap(h_found_gt[i], h_found_gt[q_index[i]]);
    }

    /* Expect all of them found */
    hash_table.Search(h_query, h_result, h_found, search_time);
    printf("4) Query finished in %.3f ms (%.3f M queries/s)\n", search_time,
           double(num_queries) / (search_time * 1000.0));
    search_success = true;
    for (int i = 0; i < num_queries; i++) {
        if (!h_found_gt[i] && h_found[i]) {
            printf("### wrong result at index %d: should be NOT FOUND\n", i);
            search_success = false;
        }
        if (h_found_gt[i] && !h_found[i]) {
            printf("### wrong result at index %d: should be FOUND\n", i);
            search_success = false;
        }
        if (h_found_gt[i] && h_found[i] && (h_result_gt[i] != h_result[i])) {
            printf("### wrong result at index %d: [%d] -> %d, but should be "
                   "%d\n",
                   i, h_query[i][0], h_result[i], h_result_gt[i]);
            search_success = false;
        }
    }
    if (search_success) {
        printf("4) Validation done\n");
    }

    load_factor = hash_table.ComputeLoadFactor(1);
    printf("The load factor is %.2f, number of buckets %d\n", load_factor,
           num_buckets);

    return 0;
}