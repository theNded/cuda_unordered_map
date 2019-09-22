#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>
#include <unordered_map>
#include "cuda_unordered_map.h"
#include "coordinate.h"

#include "custom_class.h"

void TEST_SIMPLE() {
    std::unordered_map<int, int> unordered_map;

    // insert
    std::vector<int> insert_keys = {1, 3, 5};
    std::vector<int> insert_vals = {100, 300, 500};
    for (int i = 0; i < insert_keys.size(); ++i) {
      unordered_map[insert_keys[i]] = insert_vals[i];
    }

    cuda::unordered_map<int, int> cuda_unordered_map(10);
    thrust::device_vector<int> cuda_insert_keys = insert_keys;
    thrust::device_vector<int> cuda_insert_vals = insert_vals;
    cuda_unordered_map.Insert(cuda_insert_keys, cuda_insert_vals);

    // query
    thrust::device_vector<int> cuda_query_keys(std::vector<int>({1, 2, 3, 4, 5}));
    auto cuda_query_results_ro = cuda_unordered_map.Search(cuda_query_keys);
    auto cuda_query_results_rw = cuda_unordered_map.Search_(cuda_query_keys);

    for (int i = 0; i < cuda_query_keys.size(); ++i) {
      auto iter = unordered_map.find(cuda_query_keys[i]);
      if (iter == unordered_map.end()) {
        assert(cuda_query_results_ro.second[i] == 0);
        assert(cuda_query_results_rw.second[i] == 0);
      } else {
        assert(cuda_query_results_ro.first[i] == iter->second);

        _Iterator<int, int> iterator = cuda_query_results_rw.first[i];
        // _Iterator == _Pair*
        _Pair<int, int> kv = *(thrust::device_ptr<_Pair<int, int>>(iterator));
        assert(kv.first == cuda_query_keys[i]);
        assert(kv.second == iter->second);

      }
    }

    std::cout << "TEST_SIMPLE() passed\n";
}

void TEST_6DIM_KEYS_THRUST(int key_size) {
    std::default_random_engine generator;
    std::uniform_int_distribution<int> dist(-1000, 1000);

    // generate data
    std::cout << "generating data...\n";
    std::vector<Vector6i> insert_keys(key_size);
    std::vector<int> insert_vals(key_size);
    for (int i = 0; i < key_size; ++i) {
      insert_keys[i].Random_(generator, dist);
      insert_vals[i] = i;
    }
    thrust::device_vector<Vector6i> cuda_insert_keys = insert_keys;
    thrust::device_vector<int> cuda_insert_vals = insert_vals;
    std::cout << "data generated\n";

    // cpu groundtruth
    std::cout << "generating std::unordered_map ground truth...\n";
    std::unordered_map<Vector6i, int> unordered_map;
    for (int i = 0; i < key_size; ++i) {
      unordered_map[insert_keys[i]] = insert_vals[i];
    }
    std::cout << "ground truth generated\n";

    // gpu test
    std::cout << "inserting to cuda::unordered_map...\n";
    cuda::unordered_map<Vector6i, int> cuda_unordered_map(key_size);
    cuda_unordered_map.Insert(cuda_insert_keys, cuda_insert_vals);
    std::cout << "insertion finished\n";

    // query -- all true
    std::cout << "generating query_data...\n";
    thrust::device_vector<Vector6i> cuda_query_keys(insert_keys.size());
    for (int i = 0; i < key_size; ++i) {
      if (i % 3 == 2) {
        cuda_query_keys[i] = cuda_insert_keys[i];
      } else {
        cuda_query_keys[i] = Vector6i().Random_(generator, dist);
      }
    }
    std::cout << "query data generated\n";

    std::cout << "query from cuda::unordered_map...\n";
    auto cuda_query_results = cuda_unordered_map.Search(cuda_query_keys);
    std::cout << "query results generated\n";

    std::cout << "comparing query results against ground truth...\n";
    for (int i = 0; i < cuda_query_keys.size(); ++i) {
      auto iter = unordered_map.find(cuda_query_keys[i]);
      if (iter == unordered_map.end()) {
        assert(cuda_query_results.second[i] == 0);
      } else {
        assert(cuda_query_results.first[i] == iter->second);
      }
    }

    std::cout << "TEST_6DIM_KEYS_THRUST() passed\n";
}

void TEST_6DIM_KEYS_STD(int key_size) {
    std::default_random_engine generator;
    std::uniform_int_distribution<int> dist(-1000, 1000);

    // generate data
    std::cout << "generating data...\n";
    std::vector<Vector6i> insert_keys(key_size);
    std::vector<int> insert_vals(key_size);
    for (int i = 0; i < key_size; ++i) {
      insert_keys[i].Random_(generator, dist);
      insert_vals[i] = i;
    }
    std::vector<Vector6i> cuda_insert_keys = insert_keys;
    std::vector<int> cuda_insert_vals = insert_vals;
    std::cout << "data generated\n";

    // cpu groundtruth
    std::cout << "generating std::unordered_map ground truth...\n";
    std::unordered_map<Vector6i, int> unordered_map;
    for (int i = 0; i < key_size; ++i) {
      unordered_map[insert_keys[i]] = insert_vals[i];
    }
    std::cout << "ground truth generated\n";

    // gpu test
    std::cout << "inserting to cuda::unordered_map...\n";
    cuda::unordered_map<Vector6i, int> cuda_unordered_map(key_size);
    cuda_unordered_map.Insert(cuda_insert_keys, cuda_insert_vals);
    std::cout << "insertion finished\n";

    // query -- all true
    std::cout << "generating query_data...\n";
    std::vector<Vector6i> cuda_query_keys(insert_keys.size());
    for (int i = 0; i < key_size; ++i) {
      if (i % 3 == 2) {
        cuda_query_keys[i] = cuda_insert_keys[i];
      } else {
        cuda_query_keys[i] = Vector6i().Random_(generator, dist);
      }
    }
    std::cout << "query data generated\n";

    std::cout << "query from cuda::unordered_map...\n";
    auto cuda_query_results = cuda_unordered_map.Search(cuda_query_keys);
    std::cout << "query results generated\n";

    std::cout << "comparing query results against ground truth...\n";
    for (int i = 0; i < cuda_query_keys.size(); ++i) {
      auto iter = unordered_map.find(cuda_query_keys[i]);
      if (iter == unordered_map.end()) {
        assert(cuda_query_results.second[i] == 0);
      } else {
        assert(cuda_query_results.first[i] == iter->second);
      }
    }

    std::cout << "TEST_6DIM_KEYS_STD() passed\n";
}

void TEST_COORDS(int key_size) {
    const int D = 8;
    std::default_random_engine generator;
    std::uniform_int_distribution<int> dist(-1000, 1000);

    // generate raw data (a bunch of data mimicking at::Tensor)
    std::cout << "generating data...\n";
    std::vector<int> input_coords(key_size * D);
    for (int i = 0; i < key_size * D; ++i) {
      input_coords[i] = dist(generator);
    }
    std::cout << "data generated\n";

    // convert raw data (at::Tensor) to std::vector
    // and prepare indices
    std::cout << "converting format...\n";
    std::vector<Coordinate<int, D>> insert_keys(key_size);
    std::memcpy(insert_keys.data(), input_coords.data(),
                sizeof(int) * key_size * D);
    std::vector<int> insert_vals(key_size);
    std::iota(insert_vals.begin(), insert_vals.end(), 0);

    // also make sure memcpy works correctly
    for (int i = 0; i < key_size; ++i) {
      for (int d = 0; d < D; ++d) {
        assert(input_coords[i * D + d] == insert_keys[i][d]);
      }
    }
    std::cout << "conversion finished\n";

    // cpu groundtruth
    std::cout << "generating std::unordered_map ground truth hashtable...\n";
    std::unordered_map<Coordinate<int, D>, int, CoordinateHashFunc<int, D>> unordered_map;
    for (int i = 0; i < key_size; ++i) {
      unordered_map[insert_keys[i]] = insert_vals[i];
    }
    std::cout << "ground truth generated\n";

    // gpu test
    std::cout << "inserting to cuda::unordered_map...\n";
    std::vector<Coordinate<int, D>> cuda_insert_keys = insert_keys;
    std::vector<int> cuda_insert_vals = insert_vals;
    cuda::unordered_map<Coordinate<int, D>, int> cuda_unordered_map(key_size);
    cuda_unordered_map.Insert(cuda_insert_keys, cuda_insert_vals);
    std::cout << "insertion finished\n";

    // query
    std::cout << "generating query_data...\n";
    std::vector<Coordinate<int, D>> cuda_query_keys(insert_keys.size());
    for (int i = 0; i < key_size; ++i) {
      if (i % 3 != 2) { // 2/3 is valid
        cuda_query_keys[i] = cuda_insert_keys[i];
      } else { // 1/3 is invalid
        cuda_query_keys[i] = Coordinate<int, D>::random(generator, dist);
      }
    }
    std::cout << "query data generated\n";

    std::cout << "query from cuda::unordered_map...\n";
    auto cuda_query_results = cuda_unordered_map.Search(cuda_query_keys);
    std::cout << "query results generated\n";

    std::cout << "comparing query results against ground truth...\n";
    for (int i = 0; i < cuda_query_keys.size(); ++i) {
      auto iter = unordered_map.find(cuda_query_keys[i]);
      if (iter == unordered_map.end()) {
        assert(cuda_query_results.second[i] == 0);
      } else {
        assert(cuda_query_results.first[i] == iter->second);
      }
    }

    std::cout << "TEST_COORDS() passed\n";
}

int main() {
  TEST_SIMPLE();
  TEST_6DIM_KEYS_THRUST(1000000);
  TEST_6DIM_KEYS_STD(1000000);
  TEST_COORDS(1000000);
}