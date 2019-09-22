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
#include "coordinate_indexer.cuh"

using KeyT = int;
constexpr size_t D = 7;
using ValueT = uint32_t;
using HashFunc = CoordinateHashFunc<KeyT, D>;
using KeyTD = Coordinate<KeyT, D>;

struct DataTupleCPU {
    std::vector<KeyTD> keys;
    std::vector<ValueT> values;
    std::vector<uint8_t> masks;

    void Resize(uint32_t size) {
        keys.resize(size);
        values.resize(size);
        masks.resize(size);
    }
};

struct DataTupleGPU {
    KeyT *keys = nullptr;
    ValueT *values = nullptr;
    uint8_t *masks = nullptr;
    uint32_t size;

    void Resize(uint32_t new_size) {
        Free();
        CHECK_CUDA(cudaMalloc(&keys, sizeof(KeyT) * D * new_size));
        CHECK_CUDA(cudaMalloc(&values, sizeof(ValueT) * new_size));
        CHECK_CUDA(cudaMalloc(&masks, sizeof(uint8_t) * new_size));

        size = new_size;
    }

    void Upload(const DataTupleCPU &data, bool only_keys = false) {
        assert(size == data.keys.size());
        CHECK_CUDA(cudaMemcpy(keys, data.keys.data(), sizeof(KeyT) * D * size,
                              cudaMemcpyHostToDevice));
        if (!only_keys) {
            CHECK_CUDA(cudaMemcpy(values, data.values.data(),
                                  sizeof(ValueT) * size,
                                  cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(masks, data.masks.data(),
                                  sizeof(uint8_t) * size,
                                  cudaMemcpyHostToDevice));
        }
    }

    void Download(DataTupleCPU &data) {
        assert(size == data.keys.size());
        CHECK_CUDA(cudaMemcpy(data.keys.data(), keys, sizeof(KeyT) * D * size,
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(data.values.data(), values, sizeof(ValueT) * size,
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(data.masks.data(), masks, sizeof(uint8_t) * size,
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
    std::tuple<DataTupleGPU, DataTupleGPU, DataTupleCPU> GenerateData(
            uint32_t num_queries,
            uint32_t num_valid_queries,
            int64_t seed = 1) {
        /** keys[i in 0 : hit_keys_pool_size_] = i
            keys[i in hit_keys_pool_size_ : keys_pool_size] = NOT_FOUND **/
        std::mt19937 rng(seed);

        /** prepared keys **/
        std::vector<uint32_t> index(num_queries * D);
        std::iota(index.begin(), index.end(), 0);
        std::shuffle(index.begin(), index.end(), rng);

        DataTupleCPU insert_data_cpu, query_data_cpu_gt;
        insert_data_cpu.Resize(num_valid_queries);
        query_data_cpu_gt.Resize(num_queries);

        for (int i = 0; i < num_queries; i++) {
            for (int k = 0; k < D; ++k) {
                query_data_cpu_gt.keys[i][k] = index[i * D + k];
            }
            bool flag = i < num_valid_queries;
            query_data_cpu_gt.values[i] = flag ? i : 0;
            query_data_cpu_gt.masks[i] = flag ? 1 : 0;
        }

        insert_data_cpu.keys = std::vector<KeyTD>(
                query_data_cpu_gt.keys.begin(),
                query_data_cpu_gt.keys.begin() + num_valid_queries);
        insert_data_cpu.values = std::vector<ValueT>(
                query_data_cpu_gt.values.begin(),
                query_data_cpu_gt.values.begin() + num_valid_queries);
        insert_data_cpu.masks = std::vector<uint8_t>(
                query_data_cpu_gt.masks.begin(),
                query_data_cpu_gt.masks.begin() + num_valid_queries);

        DataTupleGPU insert_data_gpu, query_data_gpu;
        insert_data_gpu.Resize(num_valid_queries);
        query_data_gpu.Resize(num_queries);

        insert_data_gpu.Upload(insert_data_cpu);
        query_data_gpu.Upload(query_data_cpu_gt, /* only keys = */ true);

        return std::make_tuple(insert_data_gpu, query_data_gpu,
                               query_data_cpu_gt);
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
};

int TestInsert() {
    const int num_valid_queries = 1 << 20;
    const int num_all_queries = 1 << 21;

    float time;
    CoordinateIndexer<D> hash_indexer(num_valid_queries);

    TestDataHelperGPU data_generator;
    auto insert_query_data_tuple =
            data_generator.GenerateData(num_all_queries, num_valid_queries);

    auto &insert_data_gpu = std::get<0>(insert_query_data_tuple);
    time = hash_indexer.Build(insert_data_gpu.keys, num_valid_queries);
    printf("1) Hash indexer in %.3f ms (%.3f M elements/s)\n", time,
           double(insert_data_gpu.size) / (time * 1000.0));
    printf("   Load factor = %f\n", hash_indexer.ComputeLoadFactor());

    auto &query_data_gpu = std::get<1>(insert_query_data_tuple);
    auto &query_data_cpu_gt = std::get<2>(insert_query_data_tuple);
    time = hash_indexer.Search(query_data_gpu.keys, query_data_gpu.values,
                               query_data_gpu.masks, num_all_queries);
    printf("2) Hash indexer searched in %.3f ms (%.3f M queries/s)\n", time,
           double(query_data_cpu_gt.keys.size()) / (time * 1000.0));

    DataTupleCPU query_data_cpu;
    query_data_cpu.Resize(query_data_gpu.size);
    query_data_gpu.Download(query_data_cpu);
    bool query_correct = data_generator.CheckQueryResult(
            query_data_cpu.values, query_data_cpu.masks,
            query_data_cpu_gt.values, query_data_cpu_gt.masks);
    if (!query_correct) return -1;

    return 0;
}

int main() {
    TestInsert();
    return 0;
}
