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
    auto cuda_query_results = cuda_unordered_map.Search(cuda_query_keys);

    for (int i = 0; i < cuda_query_keys.size(); ++i) {
      auto iter = unordered_map.find(cuda_query_keys[i]);
      if (iter == unordered_map.end()) {
        assert(cuda_query_results.second[i] == 0);
      } else {
        assert(cuda_query_results.first[i] == iter->second);
      }
    }

    std::cout << "TEST_SIMPLE() passed\n";
}

struct Vector6i {
  int x[6];

  __device__  __host__ Vector6i() {};
  __host__ Vector6i Random_(std::default_random_engine& generator,
                        std::uniform_int_distribution<int> &dist) {
    for (int i = 0; i < 6; ++i) {
      x[i] = dist(generator);
    }
    return *this;
  }

  __device__ __host__ bool operator ==(const Vector6i &other) const {
    bool res = true;
    for (int i = 0; i < 6; ++i) {
      res = res && (other.x[i] == x[i]);
    }
    return res;
  }
};

namespace std {
  template <>
  struct hash<Vector6i> {
  std::size_t operator()(const Vector6i& k) const {
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < 6; ++i) {
        h ^= k.x[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
  }
};
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

int main() {
  TEST_SIMPLE();
  TEST_6DIM_KEYS_THRUST(1000000);
  TEST_6DIM_KEYS_STD(1000000);
}