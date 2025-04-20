#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <hwy/highway.h>


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

#define DEFINE_SIMD_BINARY_OP(Symbol, expr_scalar, expr_simd)        \
struct Symbol##Op {                                                  \
    /* SIMD overload */                                              \
    template <typename V> HWY_INLINE V operator()(V A, V B) const {  \
        return (expr_simd);                                          \
    }                                                                \
    /* scalar overload */                                            \
    HWY_INLINE float operator()(float a, float b) const {            \
        return (expr_scalar);                                        \
    }                                                                \
};                                                                   \
inline nb::ndarray<nb::numpy, float, nb::ndim<1>>                    \
Symbol(nb::ndarray<float, nb::c_contig> a,                           \
    nb::ndarray<float, nb::c_contig> b) {                            \
    return binary<Symbol##Op>(a, b);                                 \
}

template <typename Op>
nb::ndarray<nb::numpy, float, nb::ndim<1>>
binary(nb::ndarray<float, nb::c_contig> a,
    nb::ndarray<float, nb::c_contig> b)
{
    const size_t N = a.shape(0);
    if (b.shape(0) != N) throw std::runtime_error("shape mismatch");
    float* A = a.data(); float* B = b.data();
    const size_t alignment = 64;
    const size_t bytes = N * sizeof(float);
    const size_t padded = ((bytes + alignment - 1) / alignment) * alignment;
    void* raw_ptr = nullptr;
#if defined(_MSC_VER)
    raw_ptr = _aligned_malloc(padded, alignment);
    if (!raw_ptr) throw std::bad_alloc();
#elif defined(__APPLE__)
    if (posix_memalign(&raw_ptr, alignment, padded) != 0)
        throw std::bad_alloc();
#else
    raw_ptr = aligned_alloc(alignment, padded);
    if (!raw_ptr) throw std::bad_alloc();
#endif
    float* C = static_cast<float*>(raw_ptr);
#if defined(_MSC_VER)
    nb::capsule deleter(C, [](void* p) noexcept { _aligned_free(p); });
#else
    nb::capsule deleter(C, [](void* p) noexcept { free(p); });
#endif

    const ScalableTag<float> d;
    const size_t L = Lanes(d);
    Op op;  size_t i = 0;
    for (; i + L <= N; i += L)
        Store(op(Load(d, A+i), Load(d, B+i)), d, C+i);
    for (; i < N; ++i)
        C[i] = op(A[i], B[i]);
    return {C, {N}, deleter};
}

DEFINE_SIMD_BINARY_OP(Add, a + b, Add(A,B));
DEFINE_SIMD_BINARY_OP(Sub, a - b, Sub(A,B));
DEFINE_SIMD_BINARY_OP(Mul, a * b, Mul(A,B));
DEFINE_SIMD_BINARY_OP(Div, a / b, Div(A,B));

inline float reduce_sum(nb::ndarray<float, nb::c_contig> a) {
    float* A = a.data();
    const size_t N = a.shape(0);
    const ScalableTag<float> d;
    auto acc = Zero(d);
    size_t i = 0, L = Lanes(d);
    for (; i + L <= N; i += L)
        acc = Add(acc, Load(d, A + i));
    float total = GetLane(SumOfLanes(d, acc));
    for (; i < N; ++i) total += A[i];
    return total;
}

inline float reduce_max(nb::ndarray<float, nb::c_contig> a) {
    float* A = a.data();
    const size_t N = a.shape(0);
    const ScalableTag<float> d;
    auto acc = Load(d, A);  // seed with first lane
    size_t i = Lanes(d), L = Lanes(d);
    for (; i + L <= N; i += L)
        acc = Max(acc, Load(d, A + i));
    float m = GetLane(MaxOfLanes(d, acc));
    for (; i < N; ++i)
        m = std::max(m, A[i]);
    return m;
}

inline float dot(nb::ndarray<float, nb::c_contig> a,
                 nb::ndarray<float, nb::c_contig> b) {
    float* A = a.data(); float* B = b.data();
    const size_t N = a.shape(0);
    if (b.shape(0) != N) throw std::runtime_error("shape mismatch");
    const ScalableTag<float> d;
    auto acc = Zero(d);
    size_t i = 0, L = Lanes(d);
    for (; i + L <= N; i += L)
        acc = Add(acc, Mul(Load(d, A + i), Load(d, B + i)));
    return GetLane(SumOfLanes(d, acc));
}

static float _dot(const float* A, const float* B, size_t N) {
    const ScalableTag<float> d;
    auto acc = Zero(d);
    size_t i = 0, L = Lanes(d);
    for (; i + L <= N; i += L)
        acc = Add(acc, Mul(Load(d, A + i), Load(d, B + i)));
    float sum = GetLane(SumOfLanes(d, acc));
    for (; i < N; ++i) sum += A[i] * B[i];
    return sum;
}

inline nb::ndarray<nb::numpy, float, nb::ndim<2>>
matmul(nb::ndarray<float, nb::c_contig, nb::ndim<2>> A,
       nb::ndarray<float, nb::c_contig, nb::ndim<2>> B) {
    size_t M = A.shape(0), K = A.shape(1);
    if (B.shape(0) != K)
        throw std::runtime_error("matmul: inner dims must match");
    size_t N = B.shape(1);

    const size_t alignment = 64;
    size_t bytes_C = M * N * sizeof(float);
    void* rawC = nullptr;
#if defined(_MSC_VER)
    rawC = _aligned_malloc(bytes_C, alignment);
    if (!rawC) throw std::bad_alloc();
#elif defined(__APPLE__)
    if (posix_memalign(&rawC, alignment, bytes_C) != 0)
        throw std::bad_alloc();
#else
    rawC = aligned_alloc(alignment, bytes_C);
    if (!rawC) throw std::bad_alloc();
#endif
    float* C = static_cast<float*>(rawC);

    // C = alpha*A*B + beta*C
    // CBLAS is column-major, but we have row-major data, so we can compute:
    // C^T = (B^T)*(A^T) which is equivalent to B*A in row-major
    // were computing: C = B*A by telling CBLAS we want: C^T = B^T*A^T
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                M, N, K, 
                alpha, 
                A.data(), K,  
                B.data(), N,   
                beta, 
                C, N);       

#if defined(_MSC_VER)
    nb::capsule deleter(C, [](void* p) noexcept { _aligned_free(p); });
#else
    nb::capsule deleter(C, [](void* p) noexcept { free(p); });
#endif
    
    return { C, { M, N }, deleter };
}

inline void register_ops(nb::module_& m) {
    m.def("add", &Add, "Element-wise add");
    m.def("sub", &Sub, "Element-wise subtract");
    m.def("mul", &Mul, "Element-wise multiply");
    m.def("div", &Div, "Element-wise divide");
    m.def("reduce_sum", &reduce_sum, "Sum of elements");
    m.def("reduce_max", &reduce_max, "Max of elements");
    m.def("dot", &dot, "Dot product");
    m.def("matmul", &matmul, "Matrix multiplication");
}

} // namespace capnhook
} // namespace HWY_NAMESPACE
} // namespace hwy
HWY_AFTER_NAMESPACE();

namespace capnhook = hwy::HWY_NAMESPACE::capnhook;
