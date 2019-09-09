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

void TEST_MINIMAL_THRUST_INSERT() {
    std::unordered_map<int, int> unordered_map;

    /** Insertion **/
    std::vector<int> insert_keys = {1, 3, 5};
    std::vector<int> insert_vals = {100, 300, 500};
    for (int i = 0; i < insert_keys.size(); ++i) {
      unordered_map[insert_keys[i]] = insert_vals[i];
    }

    cuda::unordered_map<int, int> cuda_unordered_map(10);
    thrust::device_vector<int> cuda_insert_keys = insert_keys;
    thrust::device_vector<int> cuda_insert_vals = insert_vals;
    cuda_unordered_map.Insert(cuda_insert_keys, cuda_insert_vals);

    /** Query **/
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

    std::cout << "TEST_MINIMAL_THRUST_INSERT() passed\n";
}

int main() {
  TEST_MINIMAL_THRUST_INSERT();
}