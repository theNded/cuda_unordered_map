#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <iostream>
#include <unordered_map>

template <uint8_t D>
struct Coord {
    int32_t data_[D];

    int32_t& operator[](int i) { return data_[i]; }
    const int32_t& operator[](int i) const { return data_[i]; }
};

int main() {
    std::string stra = "asdasds";
    std::string strb = "a";
    std::cout << sizeof(stra) << " " << sizeof(strb) << "\n";
    thrust::device_vector<bool> b;
    b.push_back(true);
    b.push_back(false);
    for (int i = 0; i < b.size(); ++i) {
        std::cout << b[i] << "\n";
    }

    thrust::device_vector<float> a;
    a.push_back(3.14);

    for (int i = 0; i < a.size(); ++i) {
        std::cout << a[i] << "\n";
    }

    thrust::host_vector<Coord<3>> coords_host(10);
    for (auto& coord : coords_host) {
        coord[0] = -1;
    }

    thrust::device_vector<Coord<3>> coords_device(10);
    thrust::copy(coords_host.begin(), coords_host.end(), coords_device.begin());

    // for (int i = 0; i < 10; ++i) {
    //     Coord<3> coord = coords_device[i];
    //     std::cout << coord[0] << "\n";
    // }

    std::unordered_map<int, int> hash_map;
    auto ret0 = hash_map.emplace(1, 10);
    auto ret1 = hash_map.emplace(1, 91);

    std::cout << (*ret0.first).first << " " << (*ret0.first).second << " "
              << ret0.second << "\n";
    (*ret0.first).second = -1;
    std::cout << (*ret1.first).first << " " << (*ret1.first).second << " "
              << ret1.second << "\n";
}
