#include <cstdint>
#include <random>

struct Vector6i {
    int x[6];

    __device__ __host__ Vector6i(){};
    __host__ Vector6i Random_(std::default_random_engine& generator,
                              std::uniform_int_distribution<int>& dist) {
        for (int i = 0; i < 6; ++i) {
            x[i] = dist(generator);
        }
        return *this;
    }

    __device__ __host__ bool operator==(const Vector6i& other) const {
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
}  // namespace std
