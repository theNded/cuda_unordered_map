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
#include <thrust/device_vector.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>
#include "coordinate.h"
#include "unordered_map.h"

constexpr size_t D = 7;
using KeyT = Coordinate<int32_t, D>;
using ValueT = uint32_t;
using HashFunc = CoordinateHashFunc<int32_t, D>;

struct DataTupleCPU {
    std::vector<KeyT> keys;
    std::vector<ValueT> values;
    std::vector<uint8_t> masks;

    void Resize(uint32_t size) {
        keys.resize(size);
        values.resize(size);
        masks.resize(size);
    }

    void Shuffle(int64_t seed) {
        std::mt19937 rng(seed);
        std::vector<int> q_index(keys.size());
        std::iota(q_index.begin(), q_index.end(), 0);
        std::shuffle(q_index.begin(), q_index.end(), rng);

        /* Technically this is not totally correct, as the rotated indices can
         * be swapped again elsewhere */
        for (int i = 0; i < keys.size(); i++) {
            std::swap(keys[i], keys[q_index[i]]);
            std::swap(values[i], values[q_index[i]]);
            std::swap(masks[i], masks[q_index[i]]);
        }
    }
};

struct DataTupleThrust {
    thrust::device_vector<KeyT> keys;
    thrust::device_vector<ValueT> values;
    thrust::device_vector<uint8_t> masks;

    uint32_t size;
    void Resize(uint32_t new_size) {
        Free();

        keys.resize(new_size);
        values.resize(new_size);
        masks.resize(new_size);

        size = new_size;
    }

    void Upload(const DataTupleCPU &data, bool only_keys = false) {
        assert(size == data.keys.size());
        keys = data.keys;
        if (!only_keys) {
            values = data.values;
            masks = data.masks;
        }
    }

    void Download(DataTupleCPU &data) {
        assert(size == data.keys.size());
        CHECK_CUDA(cudaMemcpy(data.keys.data(),
                              thrust::raw_pointer_cast(keys.data()),
                              sizeof(KeyT) * size, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(data.values.data(),
                              thrust::raw_pointer_cast(values.data()),
                              sizeof(ValueT) * size, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(data.masks.data(),
                              thrust::raw_pointer_cast(masks.data()),
                              sizeof(uint8_t) * size, cudaMemcpyDeviceToHost));
    }

    void Free() {
        keys.clear();
        values.clear();
        masks.clear();
    }
};

class TestDataHelperThrust {
public:
    TestDataHelperThrust(const int keys_pool_size,
                         const float hit_keys_ratio,
                         const int64_t seed = 1)
        : keys_pool_size_(keys_pool_size), seed_(seed) {
        hit_keys_pool_size_ =
                static_cast<uint32_t>(keys_pool_size_ * hit_keys_ratio);

        keys_pool_.resize(keys_pool_size_);
        values_pool_.resize(hit_keys_pool_size_);

        GenerateDataPool();
    }

    void GenerateDataPool() {
        /** keys[i in 0 : hit_keys_pool_size_] = i
            keys[i in hit_keys_pool_size_ : keys_pool_size] = NOT_FOUND **/
        std::mt19937 rng(seed_);

        std::vector<uint32_t> index(keys_pool_size_ * D);
        std::iota(index.begin(), index.end(), 0);
        std::shuffle(index.begin(), index.end(), rng);

        for (int32_t i = 0; i < keys_pool_size_; ++i) {
            for (int d = 0; d < D; ++d) {
                keys_pool_[i][d] = index[i * D + d];
            }
            if (i < hit_keys_pool_size_) {
                values_pool_[i] = i;
            }
        }
    }

    /** Return a tuple:
        @DataTupleGPU for insertion:
        - subset of query, only the 'hit' part
        @DataTupleGPU @DataTupleCPU for query:
        - all the possible queries, including 'hit' and 'miss'
        -@DataTupleGPU: keys initialized for query,
                        values and masks unintialized, reserved for return value
        -@DataTupleCPU: gt for keys, values, and masks **/
    std::tuple<DataTupleThrust, DataTupleThrust, DataTupleCPU> GenerateData(
            uint32_t num_queries, float existing_ratio) {
        uint32_t num_hit_queries =
                static_cast<uint32_t>(num_queries * existing_ratio);
        assert(num_queries <= keys_pool_size_ &&
               "num_queries > keys_pool_size_, abort");
        assert(num_hit_queries <= hit_keys_pool_size_ &&
               "num_hit_queries > hit_keys_pool_size_, abort");

        DataTupleCPU insert_data, query_data_gt;
        query_data_gt.Resize(num_queries);
        int i = 0;
        for (; i < num_hit_queries; i++) {
            query_data_gt.keys[i] = keys_pool_[i];
            query_data_gt.values[i] = values_pool_[i];
            query_data_gt.masks[i] = 1;
        }
        for (; i < num_queries; ++i) {
            query_data_gt.keys[i] = keys_pool_[i];
            query_data_gt.values[i] = 0;
            query_data_gt.masks[i] = 0;
        }

        /* insertion */
        insert_data.keys =
                std::vector<KeyT>(query_data_gt.keys.begin(),
                                  query_data_gt.keys.begin() + num_hit_queries);
        insert_data.values = std::vector<ValueT>(
                query_data_gt.values.begin(),
                query_data_gt.values.begin() + num_hit_queries);
        insert_data.masks = std::vector<uint8_t>(
                query_data_gt.masks.begin(),
                query_data_gt.masks.begin() + num_hit_queries);

        /* shuffled queries */
        insert_data.Shuffle(seed_);
        query_data_gt.Shuffle(seed_);

        DataTupleThrust insert_data_gpu, query_data_gpu;
        insert_data_gpu.Resize(num_hit_queries);
        query_data_gpu.Resize(num_queries);

        insert_data_gpu.Upload(insert_data);
        query_data_gpu.Upload(query_data_gt, /* only keys = */ true);

        return std::make_tuple(insert_data_gpu, query_data_gpu,
                               std::move(query_data_gt));
    }

    static bool CheckQueryResult(const std::vector<uint32_t> &values,
                                 const std::vector<uint8_t> &masks,
                                 const std::vector<uint32_t> &values_gt,
                                 const std::vector<uint8_t> &masks_gt) {
        int num_queries = values.size();

        for (int i = 0; i < num_queries; i++) {
            if (!masks_gt[i] && masks[i]) {
                printf("### Wrong result at index %d: should be NOT "
                       "FOUND\n",
                       i);
                return false;
            }

            if (masks_gt[i] && !masks[i]) {
                printf("### Wrong result at index %d: should be FOUND\n", i);
                return false;
            }

            if (masks_gt[i] && masks[i] && (values_gt[i] != values[i])) {
                printf("### Wrong result at index %d: %d, but should be "
                       "%d\n",
                       i, values[i], values_gt[i]);
                return false;
            }
        }
        return true;
    }

    std::vector<KeyT> keys_pool_;
    std::vector<ValueT> values_pool_;

    int keys_pool_size_;
    int hit_keys_pool_size_;

    int64_t seed_;
};

int TestInsert(TestDataHelperThrust &data_generator) {
    float time;
    CoordinateHashMap<KeyT, ValueT> hash_table(data_generator.keys_pool_size_);

    auto insert_query_data_tuple = data_generator.GenerateData(
            data_generator.keys_pool_size_ / 2, 0.4f);

    auto &insert_data_gpu = std::get<0>(insert_query_data_tuple);
    time = hash_table.Insert(insert_data_gpu.keys, insert_data_gpu.values);
    printf("1) Hash table built in %.3f ms (%.3f M elements/s)\n", time,
           double(insert_data_gpu.size) / (time * 1000.0));
    printf("   Load factor = %f\n", hash_table.ComputeLoadFactor());

    auto &query_data_gpu = std::get<1>(insert_query_data_tuple);
    auto &query_data_cpu_gt = std::get<2>(insert_query_data_tuple);
    time = hash_table.Search(query_data_gpu.keys, query_data_gpu.values,
                             query_data_gpu.masks);
    DataTupleCPU query_data_cpu;
    query_data_cpu.Resize(query_data_gpu.size);
    query_data_gpu.Download(query_data_cpu);
    printf("2) Hash table searched in %.3f ms (%.3f M queries/s)\n", time,
           double(query_data_cpu_gt.keys.size()) / (time * 1000.0));
    bool query_correct = data_generator.CheckQueryResult(
            query_data_cpu.values, query_data_cpu.masks,
            query_data_cpu_gt.values, query_data_cpu_gt.masks);
    if (!query_correct) return -1;

    return 0;
}

int TestDelete(TestDataHelperThrust &data_generator) {
    float time;
    CoordinateHashMap<KeyT, ValueT, HashFunc> hash_table(
            data_generator.keys_pool_size_);

    auto insert_query_data_tuple = data_generator.GenerateData(
            data_generator.keys_pool_size_ / 2, 1.0f);

    auto &insert_data_gpu = std::get<0>(insert_query_data_tuple);
    time = hash_table.Insert(insert_data_gpu.keys, insert_data_gpu.values);
    printf("1) Hash table built in %.3f ms (%.3f M elements/s)\n", time,
           double(insert_data_gpu.size) / (time * 1000.0));
    printf("   Load factor = %f\n", hash_table.ComputeLoadFactor());

    auto &query_data_gpu = std::get<1>(insert_query_data_tuple);
    auto &query_data_cpu_gt = std::get<2>(insert_query_data_tuple);
    time = hash_table.Search(query_data_gpu.keys, query_data_gpu.values,
                             query_data_gpu.masks);

    DataTupleCPU query_data_cpu;
    query_data_cpu.Resize(query_data_gpu.size);
    query_data_gpu.Download(query_data_cpu);
    printf("2) Hash table searched in %.3f ms (%.3f M queries/s)\n", time,
           double(query_data_cpu_gt.keys.size()) / (time * 1000.0));
    bool query_correct = data_generator.CheckQueryResult(
            query_data_cpu.values, query_data_cpu.masks,
            query_data_cpu_gt.values, query_data_cpu_gt.masks);
    if (!query_correct) return -1;

    /** Delete everything **/
    time = hash_table.Delete(query_data_gpu.keys);
    printf("3) Hash table deleted in %.3f ms (%.3f M queries/s)\n", time,
           double(query_data_gpu.size) / (time * 1000.0));
    printf("   Load factor = %f\n", hash_table.ComputeLoadFactor());

    auto query_masks_gt_after_deletion =
            std::vector<uint8_t>(query_data_cpu_gt.keys.size());
    std::fill(query_masks_gt_after_deletion.begin(),
              query_masks_gt_after_deletion.end(), 0);
    time = hash_table.Search(query_data_gpu.keys, query_data_gpu.values,
                             query_data_gpu.masks);
    printf("4) Hash table searched in %.3f ms (%.3f M queries/s)\n", time,
           double(query_data_gpu.size) / (time * 1000.0));
    query_data_gpu.Download(query_data_cpu);
    query_correct = data_generator.CheckQueryResult(
            query_data_cpu.values, query_data_cpu.masks,
            query_data_cpu_gt.values, query_masks_gt_after_deletion);

    if (!query_correct) return -1;

    return 0;
}

int TestConflict(TestDataHelperThrust &data_generator) {
    float time;
    CoordinateHashMap<KeyT, ValueT, HashFunc> hash_table(
            data_generator.keys_pool_size_);

    auto insert_query_data_tuple = data_generator.GenerateData(
            data_generator.keys_pool_size_ / 2, 1.0f);

    auto &insert_data_gpu = std::get<0>(insert_query_data_tuple);
    time = hash_table.Insert(insert_data_gpu.keys, insert_data_gpu.values);
    printf("1) Hash table built in %.3f ms (%.3f M elements/s)\n", time,
           double(insert_data_gpu.size) / (time * 1000.0));
    printf("   Load factor = %f\n", hash_table.ComputeLoadFactor());

    auto &query_data_gpu = std::get<1>(insert_query_data_tuple);
    auto &query_data_cpu_gt = std::get<2>(insert_query_data_tuple);
    time = hash_table.Search(query_data_gpu.keys, query_data_gpu.values,
                             query_data_gpu.masks);

    DataTupleCPU query_data_cpu;
    query_data_cpu.Resize(query_data_gpu.size);
    query_data_gpu.Download(query_data_cpu);
    printf("2) Hash table searched in %.3f ms (%.3f M queries/s)\n", time,
           double(query_data_cpu_gt.keys.size()) / (time * 1000.0));
    bool query_correct = data_generator.CheckQueryResult(
            query_data_cpu.values, query_data_cpu.masks,
            query_data_cpu_gt.values, query_data_cpu_gt.masks);
    if (!query_correct) return -1;

    DataTupleCPU insert_data_cpu_duplicate;
    insert_data_cpu_duplicate.Resize(insert_data_gpu.size);
    insert_data_gpu.Download(insert_data_cpu_duplicate);
    for (auto &v : insert_data_cpu_duplicate.values) {
        v += 1;
    }
    insert_data_gpu.Upload(insert_data_cpu_duplicate);

    time = hash_table.Insert(insert_data_gpu.keys, insert_data_gpu.values);
    printf("3) Hash table inserted in %.3f ms (%.3f M elements/s)\n", time,
           double(insert_data_gpu.size) / (time * 1000.0));
    printf("   Load factor = %f\n", hash_table.ComputeLoadFactor());

    time = hash_table.Search(query_data_gpu.keys, query_data_gpu.values,
                             query_data_gpu.masks);
    query_data_gpu.Download(query_data_cpu);
    printf("4) Hash table searched in %.3f ms (%.3f M queries/s)\n", time,
           double(query_data_gpu.size) / (time * 1000.0));
    query_correct = data_generator.CheckQueryResult(
            query_data_cpu.values, query_data_cpu.masks,
            query_data_cpu_gt.values, query_data_cpu_gt.masks);
    if (!query_correct) return -1;

    return 0;
}

int main() {
    const int key_value_pool_size = 1 << 20;
    const float existing_ratio = 0.6f;

    auto data_generator =
            TestDataHelperThrust(key_value_pool_size, existing_ratio);

    printf(">>> Test sequence: insert (0.5 valid) -> query\n");
    assert(!TestInsert(data_generator) && "TestInsert failed.\n");
    printf("TestInsert passed.\n");

    printf(">>> Test sequence: insert (all valid) -> query -> delete -> "
           "query\n");
    assert(!TestDelete(data_generator) && "TestDelete failed.\n");
    printf("TestDelete passed.\n");

    printf(">>> Test sequence: insert (all valid) -> query -> insert (all "
           "valid, duplicate) -> query\n");
    assert(!TestConflict(data_generator) && "TestConflict failed.\n");
    printf("TestConflict passed.\n");

    return 0;
}
