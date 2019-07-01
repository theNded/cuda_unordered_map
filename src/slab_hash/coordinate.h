//
// Created by dongw1 on 7/1/19.
//

#include <cstdint>

template <typename T, size_t D>
struct Coordinate {
private:
    T data_[D];

public:
    __device__ __host__ T& operator[](size_t i) { return data_[i]; }
    __device__ __host__ const T& operator[](size_t i) const { return data_[i]; }

    __device__ __host__ bool operator==(const Coordinate<T, D>& rhs) const {
        bool equal = true;
#pragma unroll 1
        for (size_t i = 0; i < D; ++i) {
            equal &= (data_[i] == rhs[i]);
        }
        return equal;
    }
};

template <typename T, size_t D>
struct CoordinateHashFunc {
    __device__ __host__ uint64_t operator()(const Coordinate<T, D>& key) const {
        uint64_t hash = UINT64_C(14695981039346656037);

        /** We only support 4-byte and 8-byte types **/
        using input_t = typename std::conditional<sizeof(T) == sizeof(uint32_t),
                uint32_t, uint64_t>::type;
#pragma unroll 1
        for (size_t i = 0; i < D; ++i) {
            hash ^= *((input_t*)(&key[i]));
            hash *= UINT64_C(1099511628211);
        }
        return hash;
    }
};
