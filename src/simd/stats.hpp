#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <random>
#include <hwy/contrib/sort/order.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <hwy/highway.h>
#include <hwy/contrib/sort/vqsort.h>
#include <hwy/contrib/sort/order.h>

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

/*
Given a sorted array, return the median value using VQSelect, to get the 
median value, we need to select the middle element (or the average of the two middle elements if N is even).
*/
template <typename T>
inline T medianVQSelect_(T* A, size_t N) {
    std::vector<T> copy(A, A + N);
    
    if (N % 2 == 0) {
        size_t mid = N/2;
        VQSelect(copy.data(), N, mid, SortAscending());
        T high = copy[N/2];
        
        std::vector<T> copy2(A, A + N);
        size_t midMinusOne = N/2-1;
        VQSelect(copy2.data(), N, midMinusOne, SortAscending());
        T low = copy2[N/2-1];
        
        return (low + high) / T(2);
    } else {
        size_t mid = N/2;
        VQSelect(copy.data(), N, mid, SortAscending());
        return copy[N/2];
    }
}

template <typename T>
T median(nb::ndarray<T, nb::c_contig> a) {
    T* A = a.data();
    const ScalableTag<T> d;
    const size_t L = Lanes(d);
    size_t N = a.shape(0); // non const for reallocation in pivoting

    if (N == 0) throw std::runtime_error("median: array must not be empty");
    if (N == 1) return A[0];

    // for tiny arrays (less than SIMD width), we can just do insertion sort
    if (N < L) {
        for (size_t i = 1; i < N; i++) {
            T key = A[i];
            ptrdiff_t j = static_cast<ptrdiff_t>(i) - 1;
            while (j >= 0 && A[j] > key) {
                A[j + 1] = A[j];
                j--;
            }
            A[j + 1] = key;
        }
        if (N % 2 == 0) {
            return (A[N / 2 - 1] + A[N / 2]) / T(2);
        } else {
            return A[N / 2];
        }
    }
    // VQSort(using VQSelect) works best for sizes to 32-128 (https://github.com/google/highway/blob/master/hwy/contrib/sort/README.md?utm_source=chatgpt.com)
    return medianVQSelect_<T>(A, N);
}

template <typename T>
void histogram(nb::ndarray<T, nb::c_contig> a,
               nb::ndarray<T, nb::c_contig> bins_or_edges,
               nb::ndarray<size_t, nb::c_contig> counts) {
    const size_t N = a.shape(0);
    const size_t K = bins_or_edges.shape(0);
    if (!N || !K) return;

    const bool edges = (K == counts.shape(0) + 1);
    const size_t M = edges ? K - 1 : K;

    std::vector<size_t> local(M, 0);
    const auto d = ScalableTag<T>();
    const size_t L = Lanes(d);

    auto* x = a.data();
    const T* bin = bins_or_edges.data();

    for (size_t i = 0; i + L <= N; i += L) {
        auto v = Load(d, x + i);
        for (size_t b = 0; b < M; ++b) {
            Mask<decltype(d)> m;
            if (edges) {
                m = And(Ge(v, Set(d, bin[b])), Lt(v, Set(d, bin[b + 1])));
            } else {
                auto c = Set(d, bin[b]);
                auto tol = Mul(Abs(c),
                                    Set(d, T(1e-6)));
                m = Le(Abs(Sub(v, c)), tol);
            }
            local[b] += CountTrue(d, m);
        }
    }

    for (size_t i = N - N % L; i < N; ++i) {
        for (size_t b = 0; b < M; ++b) {
            if (edges) {
                if (x[i] >= bin[b] && (b + 1 == K || x[i] < bin[b + 1]))
                    ++local[b];
            } else {
                const T tol = std::abs(bin[b]) * T(1e-6);
                if (std::abs(x[i] - bin[b]) <= tol) ++local[b];
            }
        }
    }

    std::copy(local.begin(), local.end(), counts.data());
}


template <class T>
T mode(nb::ndarray<T, nb::c_contig> a) {
  T* A = a.data();
  const size_t N = a.shape(0);
  if (N == 0) throw std::runtime_error("mode: empty array");
  if (N == 1) return A[0];

  const auto d = ScalableTag<T>();
  const size_t L = Lanes(d);

  if constexpr (std::is_integral_v<T>) {
    auto vmin = Load(d, A);
    auto vmax = vmin;
    for (size_t i = L; i + L <= N; i += L) {
      auto v = Load(d, A + i);
      vmin   = Min(vmin, v);
      vmax   = Max(vmax, v);
    }
    T lo = ReduceMin(d, vmin);
    T hi = ReduceMax(d, vmax);
    if (hi - lo <= 65535) {
      const size_t M = static_cast<size_t>(hi - lo + 1);
      std::vector<size_t> bins(M, 0);

      // SIMD histogram
      for (size_t i = 0; i + L <= N; i += L) {
        auto v = Sub(Load(d, A + i), Set(d, lo));
        for (size_t b = 0; b < M; ++b) {
          auto m = Eq(v, Set(d, static_cast<T>(b)));
          bins[b] += CountTrue(d, m);
        }
      }
      for (size_t i = N - N % L; i < N; ++i) ++bins[A[i] - lo];

      // arg-max inside vector registers
      size_t best_idx = 0, best_cnt = bins[0];
      for (size_t i = 1; i < M; ++i)
        if (bins[i] > best_cnt) { best_cnt = bins[i]; best_idx = i; }

      return static_cast<T>(lo + best_idx);
    }
  }

  std::vector<T> buf(A, A + N);
  VQSort(buf.data(), N, SortAscending());

  size_t best_cnt = 1, cur_cnt = 1;
  T      best_val = buf[0], cur_val = buf[0];

  size_t i = 1;
  for (; i + L <= N; i += L) {
    auto v = Load(d, buf.data() + i);
    auto m = Eq(v, Set(d, cur_val));
    cur_cnt += CountTrue(d, m);

    // find lanes where value changes
    for (size_t lane = 0; lane < L; ++lane) {
      T val = GetLane(v);
      if (val == cur_val) continue;
      if (cur_cnt > best_cnt ||
          (cur_cnt == best_cnt && cur_val < best_val)) {
        best_cnt = cur_cnt;
        best_val = cur_val;
      }
      cur_val = val;
      cur_cnt = 1;
    }
  }
  // tail
  for (; i < N; ++i) {
    if (buf[i] == cur_val)
      ++cur_cnt;
    else {
      if (cur_cnt > best_cnt ||
          (cur_cnt == best_cnt && cur_val < best_val)) {
        best_cnt = cur_cnt;
        best_val = cur_val;
      }
      cur_val = buf[i];
      cur_cnt = 1;
    }
  }
  if (cur_cnt > best_cnt ||
      (cur_cnt == best_cnt && cur_val < best_val))
    best_val = cur_val;

  return best_val;
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
    if (N == 1) return 0.0;

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
    if (N == 1) {
        PyErr_SetString(PyExc_AttributeError, "covMatrix: need at least two samples");
        throw nb::python_error();
    }
    
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
    if (N == 1) return 0.0;

    double C = covariance(a, b);
    double stdA = stddev(a);
    double stdB = stddev(b);
    // stop div by zero
    return (stdA == 0.0 || stdB == 0.0) ? 0.0: C / (stdA * stdB);
}

template<typename T, typename... Args>
void corrMatrix(nb::ndarray<double, nb::c_contig> corr_out, nb::ndarray<T, nb::c_contig> first, Args... args) {
    std::vector<nb::ndarray<T, nb::c_contig>> arrays;
    arrays.push_back(first);
    (arrays.push_back(args), ...); // c++17 fold

    const size_t num_arrays = arrays.size();
    const size_t N = first.shape(0);
    
    if (N == 0) throw std::runtime_error("corrMatrix: arrays must not be empty");
    if (N == 1) {
        PyErr_SetString(PyExc_AttributeError, "covMatrix: need at least two samples");
        throw nb::python_error();
    }
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