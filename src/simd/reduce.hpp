#pragma once

#include <cstddef>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <hwy/highway.h>

#include "../alloc.hpp"
#include "binary.hpp"

namespace nb = nanobind;

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace capnhook {

template <typename T>
T reduce_sum(nb::ndarray<T, nb::c_contig> a) {
    const T* A = a.data();
    size_t N = a.shape(0);
    
    if (N == 0) throw std::runtime_error("reduce_sum: zero-length input");
    if (N == 1) return A[0];
    
    const ScalableTag<T> d;
    auto acc = Zero(d);
    size_t i = 0, L = Lanes(d);
    
    if (N < L) {
        T sum = T(0);
        for (size_t j = 0; j < N; j++) {
            sum += A[j];
        }
        return sum;
    }
    
    for (; i + L <= N; i += L) {
        acc = Add(acc, Load(d, A + i));
    }
    
    T total = GetLane(SumOfLanes(d, acc));
    for (; i < N; ++i) total += A[i];
    return total;
}

template <typename T>
T reduce_min(nb::ndarray<T, nb::c_contig> a) {
    const T* A = a.data();
    size_t N = a.shape(0);
    
    if (N == 0) throw std::runtime_error("reduce_min: zero-length input");
    if (N == 1) return A[0];
    
    const ScalableTag<T> d;
    size_t L = Lanes(d);
    
    if (N < L) {
        T min_val = A[0];
        for (size_t j = 1; j < N; j++) {
            min_val = std::min(min_val, A[j]);
        }
        return min_val;
    }
    
    auto acc = Load(d, A);
    size_t i = L; 
    
    for (; i + L <= N; i += L) {
        acc = Min(acc, Load(d, A + i));
    }
    
    T m = GetLane(MinOfLanes(d, acc));
    for (; i < N; ++i) m = std::min(m, A[i]);
    return m;
}

template <typename T>
T reduce_max(nb::ndarray<T, nb::c_contig> a) {
    const T* A = a.data();
    size_t N = a.shape(0);
    
    if (N == 0) throw std::runtime_error("reduce_max: zero-length input");
    if (N == 1) return A[0]; 
    
    const ScalableTag<T> d;
    size_t L = Lanes(d);
    
    if (N < L) {
        T max_val = A[0];
        for (size_t j = 1; j < N; j++) {
            max_val = std::max(max_val, A[j]);
        }
        return max_val;
    }
    
    auto acc = Load(d, A); 
    size_t i = L; 
    
    for (; i + L <= N; i += L) {
        acc = Max(acc, Load(d, A + i));
    }
    
    T m = GetLane(MaxOfLanes(d, acc));
    for (; i < N; ++i) m = std::max(m, A[i]);
    return m;
}

template <typename T>
T reduce_prod(nb::ndarray<T, nb::c_contig> a) {
    const T* A = a.data();
    size_t N = a.shape(0);
    
    if (N == 0) throw std::runtime_error("reduce_prod: zero-length input");
    if (N == 1) return A[0]; 
    
    const ScalableTag<T> d;
    size_t L = Lanes(d);
    
    if (N < L) {
        T prod = T(1);
        for (size_t j = 0; j < N; j++) {
            prod *= A[j];
        }
        return prod;
    }
    
    auto acc = Set(d, T(1));
    size_t i = 0;
    
    for (; i + L <= N; i += L) {
        acc = Mul(acc, Load(d, A + i));
    }
    
    T product = GetLane(SumOfLanes(d, acc));
    for (; i < N; ++i) product *= A[i];
    return product;
}


template <typename T>
T reduce_mean(nb::ndarray<T, nb::c_contig> a) {
    size_t N = a.shape(0);
    return reduce_sum<T>(a) / T(N);
}

template <typename T>
T reduce_var(nb::ndarray<T, nb::c_contig> a) {
    size_t N = a.shape(0);
    T mu = reduce_mean<T>(a);
    const T* A = a.data();
    T var = T(0);
    for (size_t i = 0; i < N; ++i) {
        T d = A[i] - mu;
        var += d * d;
    }
    return var / T(N);
}

template <typename T>
T reduce_std(nb::ndarray<T, nb::c_contig> a) {
    return std::sqrt(reduce_var<T>(a));
}


template <typename T>
bool reduce_any(nb::ndarray<T, nb::c_contig> a) {
    const T* A = a.data();
    size_t N = a.shape(0);
    for (size_t i = 0; i < N; ++i) if (A[i] != T(0)) return true;
    return false;
}

template <typename T>
bool reduce_all(nb::ndarray<T, nb::c_contig> a) {
    const T* A = a.data();
    size_t N = a.shape(0);
    for (size_t i = 0; i < N; ++i) if (A[i] == T(0)) return false;
    return true;
}


template <typename T>
size_t argmax(nb::ndarray<T, nb::c_contig> a) {
    const T* A = a.data();
    size_t N = a.shape(0);
    if (N == 0) throw std::runtime_error("argmax: zero-length input");
    size_t idx = 0;
    T best = A[0];
    for (size_t i = 1; i < N; ++i) {
        if (A[i] > best) { best = A[i]; idx = i; }
    }
    return idx;
}

template <typename T>
size_t argmin(nb::ndarray<T, nb::c_contig> a) {
    const T* A = a.data();
    size_t N = a.shape(0);
    if (N == 0) throw std::runtime_error("argmin: zero-length input");
    size_t idx = 0;
    T best = A[0];
    for (size_t i = 1; i < N; ++i) {
        if (A[i] < best) { best = A[i]; idx = i; }
    }
    return idx;
}


template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<1>>
cumsum(nb::ndarray<T, nb::c_contig> a) {
    size_t N = a.shape(0);
    const T* A = a.data();
    size_t bytes = N * sizeof(T);
    void* raw = aligned_alloc64(bytes);
    T* C = static_cast<T*>(raw);
#if defined(_MSC_VER)
    nb::capsule deleter(C, [](void* p) noexcept { _aligned_free(p); });
#else
    nb::capsule deleter(C, [](void* p) noexcept { free(p); });
#endif
    T acc = T(0);
    for (size_t i = 0; i < N; ++i) {
        acc += A[i];
        C[i] = acc;
    }
    return { C, { N }, deleter };
}

template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<1>>
cumprod(nb::ndarray<T, nb::c_contig> a) {
    size_t N = a.shape(0);
    const T* A = a.data();
    size_t bytes = N * sizeof(T);
    void* raw = aligned_alloc64(bytes);
    T* C = static_cast<T*>(raw);
#if defined(_MSC_VER)
    nb::capsule deleter(C, [](void* p) noexcept { _aligned_free(p); });
#else
    nb::capsule deleter(C, [](void* p) noexcept { free(p); });
#endif
    T acc = T(1);
    for (size_t i = 0; i < N; ++i) {
        acc *= A[i];
        C[i] = acc;
    }
    return { C, { N }, deleter };
}

} // capnhook
} // HWY_NAMESPACE
} // hwy
HWY_AFTER_NAMESPACE();

namespace capnhook = hwy::HWY_NAMESPACE::capnhook;
