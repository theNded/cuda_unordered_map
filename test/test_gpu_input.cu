/*
 * Copyright 2019 Saman Ashkiani, Wei Dong
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
#include "unordered_map.h"
#include "coordinate.h"

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

struct DataTupleGPU {
    KeyT *keys = nullptr;
    ValueT *values = nullptr;
    uint8_t *masks = nullptr;
    uint32_t size;

    void Resize(uint32_t new_size) {
        Free();
        CHECK_CUDA(cudaMalloc(&keys, sizeof(KeyT) * new_size));
        CHECK_CUDA(cudaMalloc(&values, sizeof(ValueT) * new_size));
        CHECK_CUDA(cudaMalloc(&masks, sizeof(uint8_t) * new_size));

        size = new_size;
    }

    void Upload(DataTupleCPU &data, uint8_t only_keys = false) {
        assert(size == data.keys.size());
        CHECK_CUDA(cudaMemcpy(keys, data.keys.data(), sizeof(KeyT) * size,
                              cudaMemcpyHostToDevice));
        if (!only_keys) {
            CHECK_CUDA(cudaMemcpy(values, data.values.data(),
                                  sizeof(ValueT) * size,
                                  cudaMemcpyHostToDevice));
            auto data_ptr = data.masks.data();
            CHECK_CUDA(cudaMemcpy(masks, data_ptr,
                                  sizeof(uint8_t) * size,
                                  cudaMemcpyHostToDevice));
        }
    }

    void Download(DataTupleCPU &data) {
        assert(size == data.keys.size());
        CHECK_CUDA(cudaMemcpy(data.keys.data(), keys, sizeof(KeyT) * size,
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(data.values.data(), values, sizeof(ValueT) * size,
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(data.masks.data(), (void*)masks,
                              sizeof(uint8_t) * size,
                              cudaMemcpyDeviceToHost));
    }

    void Free() {
        if (keys) CHECK_CUDA(cudaFree(keys));
        if (values) CHECK_CUDA(cudaFree(values));
        if (masks) CHECK_CUDA(cudaFree(masks));
        keys = nullptr;
        values = nullptr;
        masks = nullptr;
    }
};

class TestDataHelperGPU {
public:
    TestDataHelperGPU(const int keys_pool_size,
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
    std::tuple<DataTupleGPU, DataTupleGPU, DataTupleCPU> GenerateData(
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
        insert_data.keys = std::vector<KeyT>(
                query_data_gt.keys.begin(),
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

        DataTupleGPU insert_data_gpu, query_data_gpu;
        insert_data_gpu.Resize(num_hit_queries);
        query_data_gpu.Resize(num_queries);

        insert_data_gpu.Upload(insert_data);
        query_data_gpu.Upload(query_data_gt, /* only keys = */ true);

        return std::make_tuple(insert_data_gpu, query_data_gpu,
                               std::move(query_data_gt));
    }

    static uint8_t CheckQueryResult(const std::vector<uint32_t> &values,
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

int TestInsert(TestDataHelperGPU &data_generator) {
    CudaTimer timer;
    float time;

    UnorderedMap<KeyT, ValueT> hash_table(
            data_generator.keys_pool_size_);

    auto insert_query_data_tuple = data_generator.GenerateData(
            data_generator.keys_pool_size_ / 2, 0.4f);

    auto &insert_data_gpu = std::get<0>(insert_query_data_tuple);
    timer.Start();
    hash_table.Insert(insert_data_gpu.keys, insert_data_gpu.values,
                      insert_data_gpu.size);
    time = timer.Stop();
    printf("1) Hash table built in %.3f ms (%.3f M elements/s)\n", time,
           double(insert_data_gpu.size) / (time * 1000.0));
    printf("1) Load factor = %f\n", hash_table.ComputeLoadFactor());

    auto &query_data_gpu = std::get<1>(insert_query_data_tuple);
    auto &query_data_cpu_gt = std::get<2>(insert_query_data_tuple);

    timer.Start();
    auto pair = hash_table.Search(query_data_gpu.keys, query_data_gpu.size);
    time = timer.Stop();
    printf("2) Hash table searched in %.3f ms (%.3f M queries/s)\n", time,
           double(query_data_cpu_gt.keys.size()) / (time * 1000.0));

    DataTupleCPU query_data_cpu;
    // WARNING: memory leak! fix this ds later
    query_data_gpu.values = pair.first;
    query_data_gpu.masks = pair.second;
    query_data_cpu.Resize(query_data_gpu.size);
    query_data_gpu.Download(query_data_cpu);
    
    uint8_t query_correct = data_generator.CheckQueryResult(
            query_data_cpu.values, query_data_cpu.masks,
            query_data_cpu_gt.values, query_data_cpu_gt.masks);
    if (!query_correct) return -1;

    return 0;
}

int TestRemove(TestDataHelperGPU &data_generator) {
    CudaTimer timer;
    float time;
    UnorderedMap<KeyT, ValueT, HashFunc> hash_table(
            data_generator.keys_pool_size_);

    auto insert_query_data_tuple = data_generator.GenerateData(
            data_generator.keys_pool_size_ / 2, 1.0f);

    auto &insert_data_gpu = std::get<0>(insert_query_data_tuple);

    timer.Start();
    hash_table.Insert(insert_data_gpu.keys, insert_data_gpu.values,
                      insert_data_gpu.size);
    time = timer.Stop();
    printf("1) Hash table built in %.3f ms (%.3f M elements/s)\n", time,
           double(insert_data_gpu.size) / (time * 1000.0));
    printf("1) Load factor = %f\n", hash_table.ComputeLoadFactor());

    auto &query_data_gpu = std::get<1>(insert_query_data_tuple);
    auto &query_data_cpu_gt = std::get<2>(insert_query_data_tuple);

    timer.Start();
    auto pair = hash_table.Search(query_data_gpu.keys, query_data_gpu.size);
    time = timer.Stop();
    printf("2) Hash table searched in %.3f ms (%.3f M queries/s)\n", time,
           double(query_data_cpu_gt.keys.size()) / (time * 1000.0));
   
    DataTupleCPU query_data_cpu;
    query_data_gpu.values = pair.first;
    query_data_gpu.masks = pair.second;
    query_data_cpu.Resize(query_data_gpu.size);
    query_data_gpu.Download(query_data_cpu);
    uint8_t query_correct = data_generator.CheckQueryResult(
            query_data_cpu.values, query_data_cpu.masks,
            query_data_cpu_gt.values, query_data_cpu_gt.masks);
    if (!query_correct) return -1;

    /** Remove everything **/
    timer.Start();
    hash_table.Remove(query_data_gpu.keys, query_data_gpu.size);
    time = timer.Stop();
    printf("3) Hash table deleted in %.3f ms (%.3f M queries/s)\n", time,
           double(query_data_gpu.size) / (time * 1000.0));
    printf("3) Load factor = %f\n", hash_table.ComputeLoadFactor());

    auto query_masks_gt_after_deletion =
            std::vector<uint8_t>(query_data_cpu_gt.keys.size());
    std::fill(query_masks_gt_after_deletion.begin(),
              query_masks_gt_after_deletion.end(), 0);

    timer.Start();
    pair = hash_table.Search(query_data_gpu.keys, query_data_gpu.size);
    time = timer.Stop();
    printf("4) Hash table searched in %.3f ms (%.3f M queries/s)\n", time,
           double(query_data_gpu.size) / (time * 1000.0));

    query_data_gpu.values = pair.first;
    query_data_gpu.masks = pair.second;
    query_data_gpu.Download(query_data_cpu);
    query_correct = data_generator.CheckQueryResult(
            query_data_cpu.values, query_data_cpu.masks,
            query_data_cpu_gt.values, query_masks_gt_after_deletion);

    if (!query_correct) return -1;

    return 0;
}

int TestConflict(TestDataHelperGPU &data_generator) {
    CudaTimer timer;
    float time;

    UnorderedMap<KeyT, ValueT, HashFunc> hash_table(
            data_generator.keys_pool_size_);

    auto insert_query_data_tuple = data_generator.GenerateData(
            data_generator.keys_pool_size_ / 2, 1.0f);
    auto &insert_data_gpu = std::get<0>(insert_query_data_tuple);

    timer.Start();
    hash_table.Insert(insert_data_gpu.keys, insert_data_gpu.values,
                      insert_data_gpu.size);
    time = timer.Stop();
    printf("1) Hash table built in %.3f ms (%.3f M elements/s)\n", time,
           double(insert_data_gpu.size) / (time * 1000.0));
    printf("1) Load factor = %f\n", hash_table.ComputeLoadFactor());

    auto &query_data_gpu = std::get<1>(insert_query_data_tuple);
    auto &query_data_cpu_gt = std::get<2>(insert_query_data_tuple);
    timer.Start();
    auto pair = hash_table.Search(query_data_gpu.keys, query_data_gpu.size);
    time = timer.Stop();
    printf("2) Hash table searched in %.3f ms (%.3f M queries/s)\n", time,
           double(query_data_cpu_gt.keys.size()) / (time * 1000.0));

    DataTupleCPU query_data_cpu;  
    query_data_cpu.Resize(query_data_gpu.size);
    query_data_gpu.values = pair.first;
    query_data_gpu.masks = pair.second;
    query_data_gpu.Download(query_data_cpu);

    uint8_t query_correct = data_generator.CheckQueryResult(
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

    timer.Start();
    hash_table.Insert(insert_data_gpu.keys, insert_data_gpu.values,
                      insert_data_gpu.size);
    time = timer.Stop();
    printf("3) Hash table inserted in %.3f ms (%.3f M elements/s)\n", time,
           double(insert_data_gpu.size) / (time * 1000.0));
    printf("3) Load factor = %f\n", hash_table.ComputeLoadFactor());

    timer.Start();
    pair = hash_table.Search(query_data_gpu.keys, query_data_gpu.size);
    time = timer.Stop();
    printf("4) Hash table searched in %.3f ms (%.3f M queries/s)\n", time,
           double(query_data_gpu.size) / (time * 1000.0));

    query_data_gpu.values = pair.first;
    query_data_gpu.masks = pair.second;
    query_data_gpu.Download(query_data_cpu);
    query_correct = data_generator.CheckQueryResult(
            query_data_cpu.values, query_data_cpu.masks,
            query_data_cpu_gt.values, query_data_cpu_gt.masks);
    if (!query_correct) return -1;

    return 0;
}

int main() {
    const int key_value_pool_size = 1 << 20;
    const float existing_ratio = 0.6f;
    UnorderedMap<KeyT, ValueT, HashFunc> hash_table(
            key_value_pool_size);

    auto data_generator =
            TestDataHelperGPU(key_value_pool_size, existing_ratio);

    printf(">>> Test sequence: insert (0.5 valid) -> query\n");
    assert(!TestInsert(data_generator) && "TestInsert failed.\n");
    printf("TestInsert passed.\n");

    printf(">>> Test sequence: insert (all valid) -> query -> delete -> "
           "query\n");
    assert(!TestRemove(data_generator) && "TestRemove failed.\n");
    printf("TestRemove passed.\n");

    printf(">>> Test sequence: insert (all valid) -> query -> insert (all "
           "valid, duplicate) -> query\n");
    assert(!TestConflict(data_generator) && "TestConflict failed.\n");
    printf("TestConflict passed.\n");

    return 0;
}
