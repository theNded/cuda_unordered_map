#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

struct Dummy {
    virtual void print() { std::cout << "Dummy\n"; }
};

template <size_t D>
struct Vector : public Dummy {
    Vector() : Dummy() {
        for (int i = 0; i < D; ++i) {
            data_[i] = 0;
        }
    }
    int data_[D];

    int &operator[](int i) { return data_[i]; }

    void print() override {
        for (int i = 0; i < D; ++i) {
            std::cout << data_[i] << " ";
        }
        std::cout << "\n";
    }
};

std::shared_ptr<Dummy> create(int d) {
    if (d == 0) {
        return std::make_shared<Vector<1>>();
    } else if (d == 1) {
        return std::make_shared<Vector<2>>();
    } else if (d == 2) {
        return std::make_shared<Vector<3>>();
    } else if (d == 3) {
        return std::make_shared<Vector<4>>();
    }
    return std::make_shared<Vector<5>>();
}

void insert(std::shared_ptr<Dummy> &container,
            const std::vector<int> &data,
            int d) {
    if (d == 0) {
        Vector<1> &vec = (Vector<1> &)*container;
        for (int i = 0; i <= d; ++i) {
            vec[i] = data[i];
        }
    } else if (d == 1) {
        Vector<2> &vec = (Vector<2> &)*container;
        for (int i = 0; i <= d; ++i) {
            vec[i] = data[i];
        }
    } else if (d == 2) {
        Vector<3> &vec = (Vector<3> &)*container;
        for (int i = 0; i <= d; ++i) {
            vec[i] = data[i];
        }
    } else if (d == 3) {
        Vector<4> &vec = (Vector<4> &)*container;
        for (int i = 0; i <= d; ++i) {
            vec[i] = data[i];
        }
    } else if (d == 4) {
        Vector<5> &vec = (Vector<5> &)*container;
        for (int i = 0; i <= d; ++i) {
            vec[i] = data[i];
        }
    }
}

int main() {
    std::vector<int> data({0, 1, 2, 3, 4, 5});
    for (int i = 0; i < 5; ++i) {
        auto dummy_ptr = create(i);
        dummy_ptr->print();
        insert(dummy_ptr, data, i);
        dummy_ptr->print();
    }
}
