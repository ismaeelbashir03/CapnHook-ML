#pragma once

#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <hwy/highway.h>
#include <hwy/contrib/math/math-inl.h>

#include "../alloc.hpp"

namespace nb = nanobind;

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace capnhook {

template <typename T, typename Op>
nb::ndarray<nb::numpy, T, nb::ndim<1>> unary(nb::ndarray<T, nb::c_contig> a) {
    const size_t N = a.shape(0);
    const T* A = a.data();

    const size_t bytes = N * sizeof(T);
    void* raw = aligned_alloc64(bytes);
    T* C = static_cast<T*>(raw);

#if defined(_MSC_VER)
    nb::capsule deleter(C, [](void* p) noexcept { _aligned_free(p); });
#else
    nb::capsule deleter(C, [](void* p) noexcept { free(p); });
#endif

    const ScalableTag<T> d;
    Op op;
    size_t i = 0;
    const size_t L = Lanes(d);

    for (; i + L <= N; i += L) {
        auto v = Load(d, A + i);
        Store(op(d, v), d, C + i);
    }

    for (; i < N; ++i) {
        C[i] = op(A[i]);
    }

    return { C, { N }, deleter };
}

#define DEFINE_SIMD_UNARY_OP(Symbol, expr_scalar, expr_simd)        \
struct Symbol##Op {                                                  \
    template <class D, class V>                                     \
    HWY_INLINE V operator()(D d, V v) const {                       \
        return expr_simd;                                           \
    }                                                                \
    HWY_INLINE float operator()(float x) const { return (expr_scalar); } \
    HWY_INLINE double operator()(double x) const { return (expr_scalar); } \
};                                                                   \
inline nb::ndarray<nb::numpy, float, nb::ndim<1>>                     \
Symbol(nb::ndarray<float, nb::c_contig> a) {                         \
    return unary<float, Symbol##Op>(a);                              \
}                                                                    \
inline nb::ndarray<nb::numpy, double, nb::ndim<1>>                    \
Symbol(nb::ndarray<double, nb::c_contig> a) {                        \
    return unary<double, Symbol##Op>(a);                             \
}

DEFINE_SIMD_UNARY_OP(exp, std::exp(x), hwy::HWY_NAMESPACE::Exp(d, v))
DEFINE_SIMD_UNARY_OP(log, std::log(x), hwy::HWY_NAMESPACE::Log(d, v))
DEFINE_SIMD_UNARY_OP(sqrt, std::sqrt(x), hwy::HWY_NAMESPACE::Sqrt(v))
DEFINE_SIMD_UNARY_OP(sin, std::sin(x), hwy::HWY_NAMESPACE::Sin(d, v))
DEFINE_SIMD_UNARY_OP(cos, std::cos(x), hwy::HWY_NAMESPACE::Cos(d, v))
DEFINE_SIMD_UNARY_OP(asin, std::asin(x), hwy::HWY_NAMESPACE::Asin(d, v))
DEFINE_SIMD_UNARY_OP(acos, std::acos(x), hwy::HWY_NAMESPACE::Acos(d, v))

}  // capnhook
}  // HWY_NAMESPACE
}  // hwy
HWY_AFTER_NAMESPACE();

namespace capnhook = hwy::HWY_NAMESPACE::capnhook;