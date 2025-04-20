#pragma once

#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <hwy/highway.h>

#include "../alloc.hpp"

namespace nb = nanobind;

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace capnhook {

template <typename T, typename Op>
nb::ndarray<nb::numpy, T, nb::ndim<1>> binary(nb::ndarray<T, nb::c_contig> a,
                     nb::ndarray<T, nb::c_contig> b) {
    const size_t N = a.shape(0);
    if (b.shape(0) != N) throw std::runtime_error("shape mismatch");
    const T* A = a.data();
    const T* B = b.data();

    const size_t bytes = N * sizeof(T);
    void* raw = aligned_alloc64(bytes);
    T* C = static_cast<T*>(raw);

#if defined(_MSC_VER)
    nb::capsule deleter(C, [](void* p) noexcept { _aligned_free(p); });
#else
    nb::capsule deleter(C, [](void* p) noexcept { free(p); });
#endif

    const ScalableTag<T> d;
    const size_t L = Lanes(d);
    Op op;
    size_t i = 0;

    for (; i + L <= N; i += L) {
        auto va = Load(d, A + i);
        auto vb = Load(d, B + i);
        auto vc = op(va, vb);
        Store(vc, d, C + i);
    }
    for (; i < N; ++i) {
        C[i] = op(A[i], B[i]);
    }

    return nb::ndarray<nb::numpy, T, nb::ndim<1>>(C, { N }, deleter);
}

#define DEFINE_SIMD_BINARY_OP(Symbol, expr_scalar, expr_simd)                      \
struct Symbol##Op {                                                                  \
    template <typename V> HWY_INLINE V operator()(V a, V b) const {                  \
        return (expr_simd);                                                          \
    }                                                                                \
    HWY_INLINE float  operator()(float  a, float  b) const { return (expr_scalar); }  \
    HWY_INLINE double operator()(double a, double b) const { return (expr_scalar); }  \
};                                                                                   \
inline nb::ndarray<nb::numpy, float, nb::ndim<1>>                                    \
Symbol(nb::ndarray<float,  nb::c_contig> a,                                          \
       nb::ndarray<float,  nb::c_contig> b) {                                        \
    return binary<float,  Symbol##Op>(a, b);                                          \
}                                                                                    \
inline nb::ndarray<nb::numpy, double, nb::ndim<1>>                                   \
Symbol(nb::ndarray<double, nb::c_contig> a,                                          \
       nb::ndarray<double, nb::c_contig> b) {                                        \
    return binary<double, Symbol##Op>(a, b);                                          \
}

DEFINE_SIMD_BINARY_OP(add, a + b, Add(a, b))
DEFINE_SIMD_BINARY_OP(sub, a - b, Sub(a, b))
DEFINE_SIMD_BINARY_OP(mul, a * b, Mul(a, b))
DEFINE_SIMD_BINARY_OP(div, a / b, Div(a, b))

} // capnhook
} // HWY_NAMESPACE
} // hwy
HWY_AFTER_NAMESPACE();

namespace capnhook = hwy::HWY_NAMESPACE::capnhook;
