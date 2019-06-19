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
#include <thread>
#include <vector>
#include "../../../../../usr/include/c++/7/chrono"
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
    const auto f = [](const KeyT& key) { return key * 10; };

    const int num_insertions = num_elems / 5;
    std::vector<KeyT> h_key(num_insertions);
    std::vector<ValueT> h_value(num_insertions);
    const int64_t seed = 1;
    std::mt19937 rng(seed);
    std::vector<uint32_t> index(num_insertions);
    std::iota(index.begin(), index.end(), 0);
    std::shuffle(index.begin(), index.end(), rng);
    for (int32_t i = 0; i < index.size(); i++) {
        h_key[i] = index[i];
        h_value[i] = f(h_key[i]);
    }

    /******* Instantiate hash table ********/
    printf("num elems: %d, num buckets: %d -- num insertions: %d\n", num_elems,
           num_buckets, num_insertions);
    GpuHashTable<KeyT, ValueT> hash_table(num_elems, num_buckets, 0, seed);

    /****** Insert and query ********/
    float build_time =
            hash_table.Insert(h_key.data(), h_value.data(), num_insertions);
    printf("1) Insert finished in %.3f ms (%.3f M elements/s)\n", build_time,
           double(num_insertions) / (build_time * 1000.0));

    std::vector<ValueT> h_result(num_insertions);
    float search_time =
            hash_table.Search(h_key.data(), h_result.data(), num_insertions);
    printf("2) Query finished in %.3f ms (%.3f M queries/s)\n", search_time,
           double(num_insertions) / (search_time * 1000.0));

    bool search_success = true;
    for (int i = 0; i < num_insertions; i++) {
        if (f(h_key[i]) != h_result[i]) {
            printf("### Result at index %d: [%d] -> %d, expected: %d\n", i,
                   h_key[i], h_result[i], f(h_key[i]));
            search_success = false;
        }
    }
    if (search_success) {
        printf("2) Validation done\n");
    }
    double load_factor = hash_table.measureLoadFactor(1);
    printf("The load factor is %.2f, number of buckets %d\n", load_factor,
           num_buckets);

    /** Disturb the value **/
    for (auto& v : h_value) {
        v += 1;
    }

    std::this_thread::sleep_for(std::chrono::seconds(2));

    /** Insert again **/
    build_time =
            hash_table.Insert(h_key.data(), h_value.data(), num_insertions);
    printf("3) Insert finished in %.3f ms (%.3f M elements/s)\n", build_time,
           double(num_insertions) / (build_time * 1000.0));

    search_time =
            hash_table.Search(h_key.data(), h_result.data(), num_insertions);
    printf("4) Query finished in %.3f ms (%.3f M queries/s)\n", search_time,
           double(num_insertions) / (search_time * 1000.0));

    search_success = true;
    for (int i = 0; i < num_insertions; i++) {
        /* We expect nothing is changed */
        if (f(h_key[i]) != h_result[i]) {
            printf("### Result at index %d: [%d] -> %d, expected: %d\n", i,
                   h_key[i], h_result[i], f(h_key[i]));
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