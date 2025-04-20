#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "../alloc.hpp"

#ifdef USE_ACCELERATE
  #include <Accelerate/Accelerate.h>   
#else
  #include <cblas.h>                  
#endif

namespace nb = nanobind;

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace capnhook {

template <typename T>
T dot(nb::ndarray<T, nb::c_contig> a, nb::ndarray<T, nb::c_contig> b) {
    const size_t N = a.shape(0);
    
    if (b.shape(0) != N) {
        throw std::runtime_error("dot: vectors must have the same length");
    }
    
    const T* A = a.data();
    const T* B = b.data();
    
    T result = 0;
    
    if constexpr (std::is_same_v<T, float>) {
        result = cblas_sdot(N, A, 1, B, 1);
    } else if constexpr (std::is_same_v<T, double>) {
        result = cblas_ddot(N, A, 1, B, 1);
    }
    
    return result;
}

template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<2>> matmul(nb::ndarray<T, nb::c_contig, nb::ndim<2>> A,
                      nb::ndarray<T, nb::c_contig, nb::ndim<2>> B) {
    size_t M = A.shape(0), K = A.shape(1),
           K2 = B.shape(0), N = B.shape(1);
    if (K2 != K) throw std::runtime_error("matmul: inner dims must match");

    size_t bytes = M * N * sizeof(T);
    void* raw = aligned_alloc64(bytes);
    T* C   = static_cast<T*>(raw);

    T alpha = T(1), beta = T(0);
    // row‑major
    if constexpr (std::is_same_v<T, float>) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, alpha,
                    A.data(), K,
                    B.data(), N,
                    beta,
                    C, N);
    } else {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, alpha,
                    A.data(), K,
                    B.data(), N,
                    beta,
                    C, N);
    }

#if defined(_MSC_VER)
    nb::capsule deleter(C, [](void* p) noexcept { _aligned_free(p); });
#else
    nb::capsule deleter(C, [](void* p) noexcept { free(p); });
#endif
    return { C, { M, N }, deleter };
}

template <typename T>
T trace(nb::ndarray<T, nb::c_contig, nb::ndim<2>> A) {
    size_t M = A.shape(0), N = A.shape(1);
    size_t n = std::min(M, N);
    const T* data = A.data();
    T sum = T(0);
    for (size_t i = 0; i < n; ++i)
        sum += data[i * N + i];
    return sum;
}

template <typename T>
T norm(nb::ndarray<T, nb::c_contig> a) {
    return std::sqrt(dot(a, a));
}

} // capnhook
} // HWY_NAMESPACE
} // hwy
HWY_AFTER_NAMESPACE();

namespace capnhook = hwy::HWY_NAMESPACE::capnhook;