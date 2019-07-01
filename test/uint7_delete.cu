/*
 * Copyright 2019 Saman Ashkiani
 *
 * findEmptyPerWarpLicensed under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance with the License.
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

    /** Hash table meta data **/
    uint32_t num_keys = 1 << 20;

    float expected_chain = 0.6f;
    uint32_t num_elems_per_bucket = 15;
    uint32_t expected_elems_per_bucket = expected_chain * num_elems_per_bucket;
    uint32_t num_buckets = (num_keys + expected_elems_per_bucket - 1) /
                           expected_elems_per_bucket;

    /** Generate key-value pairs for building **/
    float existing_ratio = 0.6f;
    uint32_t num_queries = num_keys;
    auto num_elements = 2 * num_keys;

    using KeyT = uint32_t;
    constexpr size_t D = 7;
    using ValueT = uint32_t;
    using HashFunc = CoordinateHashFunc<KeyT, D>;
    using KeyTD = Coordinate<KeyT, D>;

    std::vector<KeyTD> h_key(num_elements);
    std::vector<ValueT> h_value(num_elements);

    std::vector<KeyTD> h_query(num_queries);
    std::vector<uint8_t> h_found(num_queries);
    std::vector<ValueT> h_result(num_queries);

    std::vector<uint8_t> h_found_gt(num_queries);
    std::vector<ValueT> h_result_gt(num_queries);

    const int64_t seed = 1;
    std::mt19937 rng(seed);
    std::vector<uint32_t> index(num_elements * D);
    std::iota(index.begin(), index.end(), 0);
    std::shuffle(index.begin(), index.end(), rng);

    for (int32_t i = 0; i < num_elements; ++i) {
        for (int d = 0; d < D; ++d) {
            h_key[i][d] = index[i * D + d];
        }
        h_value[i] = i;
    }

    /** Generate key-value pairs for query **/
    uint32_t num_existing = static_cast<uint32_t>(existing_ratio * num_queries);

    for (int i = 0; i < num_existing; i++) {
        h_query[i] = h_key[num_keys - 1 - i];
        h_result_gt[i] = h_value[num_keys - 1 - i];
        h_found_gt[i] = 1;
    }

    for (int i = 0; i < (num_queries - num_existing); i++) {
        h_query[num_existing + i] = h_key[num_keys + i];
        // h_result_gt[num_existing + i] = SEARCH_NOT_FOUND;
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

    /** Instantiate hash table **/
    CoordinateHashMap<KeyT, D, ValueT, HashFunc> hash_table(num_elements,
                                                            num_buckets, 0);

    printf("0) num_keys = %d, num_buckets = %d\n", num_keys, num_buckets);
    float build_time = 0;
    std::vector<KeyTD> keys(h_key.begin(), h_key.begin() + num_keys);
    std::vector<ValueT> values(h_value.begin(), h_value.begin() + num_keys);
    hash_table.Insert(keys, values, build_time);
    printf("1) Hash table built in %.3f ms (%.3f M elements/s)\n", build_time,
           double(num_keys) / (build_time * 1000.0));

    float search_time = 0;
    hash_table.Search(h_query, h_result, h_found, search_time);
    printf("2) Hash table (existing ratio %.2f) searched in %.3f ms (%.3f M "
           "queries/s)\n",
           existing_ratio, search_time,
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

    /* Delete everything */
    float delete_time = 0;
    hash_table.Delete(keys, delete_time);
    printf("3) Hash table deleted in %.3f ms (%.3f M queries/s)\n", delete_time,
           double(num_queries) / (delete_time * 1000.0));

    hash_table.Search(h_query, h_result, h_found, search_time);
    printf("4) Hash table (existing ratio %.2f) searched in %.3f ms (%.3f M "
           "queries/s)\n",
           existing_ratio, search_time,
           double(num_queries) / (search_time * 1000.0));
    search_success = true;
    for (int i = 0; i < num_queries; i++) {
        if (h_found[i]) {
            printf("### wrong result at index %d: should be NOT FOUND\n", i);
            search_success = false;
        }
    }
    if (search_success) {
        printf("4) Validation done\n");
    }
    load_factor = hash_table.ComputeLoadFactor(1);
    printf("The load factor is %.2f, number of buckets %d\n", load_factor,
           num_buckets);

    /* Insert everything back */
    hash_table.Insert(keys, values, build_time);
    printf("5) Hash table built in %.3f ms (%.3f M elements/s)\n", build_time,
           double(num_keys) / (build_time * 1000.0));
    hash_table.Search(h_query, h_result, h_found, search_time);
    printf("6) Hash table (existing ratio %.2f) searched in %.3f ms (%.3f M "
           "queries/s)\n",
           existing_ratio, search_time,
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
        printf("6) Validation done\n");
    }

    load_factor = hash_table.ComputeLoadFactor(1);
    printf("The load factor is %.2f, number of buckets %d\n", load_factor,
           num_buckets);

    return 0;
}