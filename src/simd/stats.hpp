#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <hwy/contrib/sort/order.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <hwy/highway.h>
#include <hwy/contrib/sort/vqsort.h>
#include <random>
#include "hwy/contrib/sort/order.h"

namespace nb = nanobind;

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace capnhook {

template <typename T>
T mean(nb::ndarray<T, nb::c_contig> a) {
    T* A = a.data();
    const size_t N = a.shape(0);
    if (N == 0) throw std::runtime_error("mean: array must not be empty");
    if (N == 1) return A[0];

    const ScalableTag<T> d;
    size_t L = Lanes(d);

    // fallback for small arrays (less than SIMD width)
    if (N < L) {
        T sum = T(0);
        for (size_t j = 0; j < N; j++) {
            sum += A[j];
        }
        return sum / T(N);
    }

    auto acc = Zero(d); // vector to store the sum
    size_t i = 0;
    for (; i + L <= N; i += L) {
        acc = Add(acc, Load(d, A + i));
    }
    T total = ReduceSum(d, acc); // get total sum
    // add remaining elements
    for (; i < N; ++i) total += A[i]; 
    return total / T(N);
}


template <typename T>
inline T medianVQSort_(T* A, size_t N) {
    VQSort(A, N, SortAscending());
    // get the middle element
    return A[N / 2];
}

template<typename T>
inline T inRegisterMedian(T* A, size_t N) {
    alignas(64) T buffer[64];
    for (size_t i = 0; i < N; ++i) {
        buffer[i] = A[i];
    }
    VQSort(buffer, N, SortAscending());
    // get the middle element
    return buffer[N / 2];
}

template <typename T>
T median(nb::ndarray<T, nb::c_contig> a) {
    T* A = a.data();
    const ScalableTag<T> d;
    const size_t L = Lanes(d);
    size_t N = a.shape(0); // non const for reallocation in pivoting

    if (N == 0) throw std::runtime_error("median: array must not be empty");
    if (N == 1) return A[0];

    // for tiny arrays (less than SIMD width), we can put them in registers and sort (no branches/memory)
    if (N < L) return inRegisterMedian(A, N);
    // VQSort works best for sizes to ~64 (https://github.com/google/highway/blob/master/hwy/contrib/sort/README.md?utm_source=chatgpt.com)
    if (N <= 64) return medianVQSort_<T>(A, N);

    // for larger arrays we do SIMD pivot search
    std::uniform_int_distribution<size_t> dist(0, N - 1);
    std::mt19937_64 rng;
    size_t left = 0, right = N;
    size_t k = N / 2; 

  
    while (true) {
        T pivot = A[dist(rng)];  // random pivot
        // temp buffers on stack (could be static/global if N large)
        T* lo = new T[N];
        T* hi = new T[N];
        size_t lo_sz = 0, hi_sz = 0;
        
        // simd partitioning
        for (size_t i = 0; i < N; i += L) {
            auto v = Load(d, A + i);
            auto mask = Lt(v,  Set(d, pivot));
            lo_sz += CompressStore(v, mask, d, lo + lo_sz);
            hi_sz += CompressStore(v, Not(mask), d, hi + hi_sz);
        }

        if (k < lo_sz) {
            // median in lo
            delete[] hi;
            N = lo_sz;  
            A = lo;
        } else if (k >= lo_sz + 1) {
            // median in hi
            k -= lo_sz + 1;
            delete[] lo;
            N = hi_sz;  
            A = hi;
        } else {
            // pivot is the median
            delete[] lo;
            delete[] hi;
            return pivot;
        }
    }
}

template<typename T>
void histogram(nb::ndarray<T, nb::c_contig> a, nb::ndarray<T, nb::c_contig> values, nb::ndarray<size_t, nb::c_contig> counts) {
    T* A = a.data();
    const size_t N = a.shape(0);
    if (N == 0) return;

    const T* bins = values.data();
    size_t* cnts = counts.data();
    const size_t M = values.shape(0);

    // get bins
    std::vector<decltype(Load(ScalableTag<T>(), bins))> bin_vecs(M);
    for (size_t b = 0; b < M; ++b) {
        bin_vecs[b] = Set(ScalableTag<T>(), bins[b]);
    }

    std::vector<size_t> local_cnt(M, 0);

    const auto d = ScalableTag<T>();
    const size_t L = Lanes(d);

    // SIMD histogram
    for (size_t i = 0; i + L <= N; i += L) {
        const auto v = Load(d, A + i);

        for (size_t b = 0; b < M; ++b) {
        // mask the equal bins, extract bits, popcount
        auto m    = Eq(v, bin_vecs[b]);    
        uint64_t mask = MaskToBits(m);
        local_cnt[b] += __builtin_popcountll(mask);
        }
    }

    // remaining elements
    for (size_t i = N - (N % L); i < N; ++i) {
        for (size_t b = 0; b < M; ++b) {
          if (A[i] == bins[b]) ++local_cnt[b];
        }
    }
    
    // add to out
    for (size_t b = 0; b < M; ++b) cnts[b] = local_cnt[b];
}

template<typename T>
T mode(nb::ndarray<T, nb::c_contig> a) {
    T* A = a.data();
    const size_t N = a.shape(0);
    if (N == 0) throw std::runtime_error("mode: array must not be empty");
    if (N == 1) return A[0];

    const ScalableTag<T> d;
    size_t L = Lanes(d);

    if (N < L) {
        size_t counts[N]; // list where index is index of array and value is count
        for (size_t j = 0; j < N; j++) {
            counts[j]++;
        }
        T mode = A[0];
        size_t max_count = 0;
        for (size_t j = 0; j < N; j++) {
            if (counts[j] > max_count) {
                max_count = counts[j];
                mode = A[j];
            }
        }
        return mode;
    }

    // SIMD mode
    std::vector<T> bins(N);
    std::vector<size_t> counts(N);
    histogram(a, nb::ndarray<T, nb::c_contig>(bins.data(), {N}), nb::ndarray<size_t, nb::c_contig>(counts.data(), {N}));
    
    int i = 0;
    T mode = bins[0];
    size_t max_count = counts[0];
    for (; i + L <= N; i += L) {
        const auto v = Load(d, A + i);
        const auto m = Load(d, counts.data() + i);
        auto mask = Gt(m, Set(d, max_count));
        uint64_t mask_bits = MaskToBits(mask);
        if (mask_bits) {
            // get the index of the first set bit
            size_t index = __builtin_ctzll(mask_bits);
            max_count = counts[index];
            mode = bins[index];
        }
    }
    // remaining elements
    for (; i < N; ++i) {
        if (counts[i] > max_count) {
            max_count = counts[i];
            mode = bins[i];
        }
    }
    return mode;
}

template<typename T>
T variance(nb::ndarray<T, nb::c_contig> a) {
    T* A = a.data();
    const size_t N = a.shape(0);
    if (N == 0) throw std::runtime_error("variance: array must not be empty");
    if (N == 1) return 0;

    const ScalableTag<T> d;
    size_t L = Lanes(d);

    // fallback for small arrays (less than SIMD width)
    if (N < L) {
        T mean_val = mean(a);
        T sum = 0;
        for (size_t j = 0; j < N; j++) {
            sum += (A[j] - mean_val) * (A[j] - mean_val);
        }
        return sum / T(N - 1);
    }

    // SIMD variance
    T mean_val = mean(a);
    auto acc = Zero(d); // vector to store the sum of squares
    size_t i = 0;
    for (; i + L <= N; i += L) {
        auto v = Load(d, A + i);
        auto diff = Sub(v, Set(d, mean_val));
        acc = Add(acc, Mul(diff, diff));
    }
    T total_sum = ReduceSum(d, acc); // get total sum of squares
    // add remaining elements
    for (; i < N; ++i) total_sum += (A[i] - mean_val) * (A[i] - mean_val); 
    return total_sum / T(N - 1);
}

template<typename T>
T stddev(nb::ndarray<T, nb::c_contig> a) {
    return std::sqrt(variance(a));
}

// get co-variance matrix
template<typename T>
double covariance(nb::ndarray<T, nb::c_contig> a, nb::ndarray<T, nb::c_contig> b) {
    T* A = a.data();
    T* B = b.data();
    const size_t N = a.shape(0);
    if (N == 0) throw std::runtime_error("covariance: array must not be empty");
    if (N != b.shape(0)) throw std::runtime_error("covariance: arrays must have same size");

    double C = 0;
    const ScalableTag<T> d;
    size_t L = Lanes(d);

    // fallback for small arrays (less than SIMD width)
    if (N < L) {
        for (size_t i = 0; i < N; i++) {
            C += (A[i] - mean(a)) * (B[i] - mean(b));
        }
        return C / (N - 1);
    }

    // SIMD covariance
    auto acc = Zero(d); // vector to store the sum of products
    size_t i = 0;
    auto a_mean = mean(a);
    auto b_mean = mean(b);
    for (; i + L <= N; i += L) {
        auto vA = Load(d, A + i);
        auto vB = Load(d, B + i);
        auto diffA = Sub(vA, Set(d, a_mean));
        auto diffB = Sub(vB, Set(d, b_mean));
        acc = Add(acc, Mul(diffA, diffB));
    }
    double total_sum = ReduceSum(d, acc); // get total sum of products
    // add remaining elements
    for (; i < N; ++i) {
        total_sum += (A[i] - a_mean) * (B[i] - b_mean);
    }
    return total_sum / (N - 1);
}

template<typename T, typename... Args>
void covMatrix(nb::ndarray<double, nb::c_contig> cov_out, nb::ndarray<T, nb::c_contig> first, Args... args) {
    std::vector<nb::ndarray<T, nb::c_contig>> arrays;
    arrays.push_back(first);
    (arrays.push_back(args), ...); // c++17 fold

    const size_t num_arrays = arrays.size();
    const size_t N = first.shape(0);
    
    if (N == 0) throw std::runtime_error("covMatrix: arrays must not be empty");
    for (size_t i = 1; i < num_arrays; i++) {
        if (arrays[i].shape(0) != N) 
            throw std::runtime_error("covMatrix: all arrays must have the same length");
    }
    
    if (cov_out.ndim() != 2 || 
    cov_out.shape(0) != num_arrays || 
    cov_out.shape(1) != num_arrays) {
        throw std::runtime_error("covMatrix: output matrix must be NxN where N is the number of input arrays");
    }
    
    double* C = cov_out.data();
    
    for (size_t i = 0; i < num_arrays; i++) {
        for (size_t j = 0; j < num_arrays; j++) {
            // C[i,j] = cov(arrays[i], arrays[j])
            C[i * num_arrays + j] = covariance(arrays[i], arrays[j]);
        }
    }
}

template<typename T>
double correlation(nb::ndarray<T, nb::c_contig> a, nb::ndarray<T, nb::c_contig> b) {
    T* A = a.data();
    T* B = b.data();
    const size_t N = a.shape(0);
    if (N == 0) throw std::runtime_error("correlation: array must not be empty");
    if (N != b.shape(0)) throw std::runtime_error("correlation: arrays must have same size");

    double C = covariance(a, b);
    double stdA = stddev(a);
    double stdB = stddev(b);
    return C / (stdA * stdB);
}

template<typename T, typename... Args>
void corrMatrix(nb::ndarray<double, nb::c_contig> corr_out, nb::ndarray<T, nb::c_contig> first, Args... args) {
    std::vector<nb::ndarray<T, nb::c_contig>> arrays;
    arrays.push_back(first);
    (arrays.push_back(args), ...); // c++17 fold

    const size_t num_arrays = arrays.size();
    const size_t N = first.shape(0);
    
    if (N == 0) throw std::runtime_error("corrMatrix: arrays must not be empty");
    for (size_t i = 1; i < num_arrays; i++) {
        if (arrays[i].shape(0) != N) 
            throw std::runtime_error("corrMatrix: all arrays must have the same length");
    }
    
    if (corr_out.ndim() != 2 || 
    corr_out.shape(0) != num_arrays || 
    corr_out.shape(1) != num_arrays) {
        throw std::runtime_error("corrMatrix: output matrix must be NxN where N is the number of input arrays");
    }
    
    double* C = corr_out.data();
    
    for (size_t i = 0; i < num_arrays; i++) {
        for (size_t j = 0; j < num_arrays; j++) {
            // C[i,j] = corr(arrays[i], arrays[j])
            C[i * num_arrays + j] = correlation(arrays[i], arrays[j]);
        }
    }
}

} // capnhook
} // HWY_NAMESPACE
} // hwy
HWY_AFTER_NAMESPACE();

namespace capnhook = hwy::HWY_NAMESPACE::capnhook;