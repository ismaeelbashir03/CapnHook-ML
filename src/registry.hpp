#pragma once

#include <nanobind/nanobind.h>
#include "simd/binary.hpp"
#include "simd/unary.hpp"
#include "simd/reduce.hpp"
#include "simd/linalg.hpp"
#include "simd/stats.hpp"

namespace registry {

template <typename T>
void register_ops(nanobind::module_& m) {
      using namespace capnhook;
    
      // binary operations
      m.def("add", static_cast<nb::ndarray<nb::numpy, T, nb::ndim<1>> (*)(nb::ndarray<T, nb::c_contig>, nb::ndarray<T, nb::c_contig>)>(&add),
            "Element-wise addition");
      m.def("sub", static_cast<nb::ndarray<nb::numpy, T, nb::ndim<1>> (*)(nb::ndarray<T, nb::c_contig>, nb::ndarray<T, nb::c_contig>)>(&sub),
            "Element-wise subtraction");
      m.def("mul", static_cast<nb::ndarray<nb::numpy, T, nb::ndim<1>> (*)(nb::ndarray<T, nb::c_contig>, nb::ndarray<T, nb::c_contig>)>(&mul),
            "Element-wise multiplication");
      m.def("div", static_cast<nb::ndarray<nb::numpy, T, nb::ndim<1>> (*)(nb::ndarray<T, nb::c_contig>, nb::ndarray<T, nb::c_contig>)>(&div),
            "Element-wise division");
      
      // unary operations
      m.def("exp", static_cast<nb::ndarray<nb::numpy, T, nb::ndim<1>> (*)(nb::ndarray<T, nb::c_contig>)>(&exp),
            "Element-wise exponential");
      m.def("log", static_cast<nb::ndarray<nb::numpy, T, nb::ndim<1>> (*)(nb::ndarray<T, nb::c_contig>)>(&log),
            "Element-wise natural logarithm");
      m.def("sqrt", static_cast<nb::ndarray<nb::numpy, T, nb::ndim<1>> (*)(nb::ndarray<T, nb::c_contig>)>(&sqrt),
            "Element-wise square root");
      m.def("sin", static_cast<nb::ndarray<nb::numpy, T, nb::ndim<1>> (*)(nb::ndarray<T, nb::c_contig>)>(&sin),
            "Element-wise sine");
      m.def("cos", static_cast<nb::ndarray<nb::numpy, T, nb::ndim<1>> (*)(nb::ndarray<T, nb::c_contig>)>(&cos),
            "Element-wise cosine");
      m.def("asin", static_cast<nb::ndarray<nb::numpy, T, nb::ndim<1>> (*)(nb::ndarray<T, nb::c_contig>)>(&asin),
            "Element-wise arcsine");
      m.def("acos", static_cast<nb::ndarray<nb::numpy, T, nb::ndim<1>> (*)(nb::ndarray<T, nb::c_contig>)>(&acos),
            "Element-wise arccosine");
      
      // reduction operations 
      m.def("reduce_sum", static_cast<T (*)(nb::ndarray<T, nb::c_contig>)>(&reduce_sum),
            "Sum reduction");
      m.def("reduce_prod", static_cast<T (*)(nb::ndarray<T, nb::c_contig>)>(&reduce_prod),
            "Product reduction");
      m.def("reduce_min", static_cast<T (*)(nb::ndarray<T, nb::c_contig>)>(&reduce_min),
            "Minimum value");
      m.def("reduce_max", static_cast<T (*)(nb::ndarray<T, nb::c_contig>)>(&reduce_max),
            "Maximum value");
      m.def("reduce_mean", static_cast<T (*)(nb::ndarray<T, nb::c_contig>)>(&reduce_mean),
            "Mean value");
      m.def("reduce_var", static_cast<T (*)(nb::ndarray<T, nb::c_contig>)>(&reduce_var),
            "Variance");
      m.def("reduce_std", static_cast<T (*)(nb::ndarray<T, nb::c_contig>)>(&reduce_std),
            "Standard deviation");
      m.def("reduce_any", static_cast<bool (*)(nb::ndarray<T, nb::c_contig>)>(&reduce_any),
            "Returns true if any element is non-zero");
      m.def("reduce_all", static_cast<bool (*)(nb::ndarray<T, nb::c_contig>)>(&reduce_all),
            "Returns true if all elements are non-zero");
      
      // index operations
      m.def("argmax", static_cast<size_t (*)(nb::ndarray<T, nb::c_contig>)>(&argmax),
            "Index of maximum value");
      m.def("argmin", static_cast<size_t (*)(nb::ndarray<T, nb::c_contig>)>(&argmin),
            "Index of minimum value");
      
      // cumulative operations
      m.def("cumsum", static_cast<nb::ndarray<nb::numpy, T, nb::ndim<1>> (*)(nb::ndarray<T, nb::c_contig>)>(&cumsum),
            "Cumulative sum");
      m.def("cumprod", static_cast<nb::ndarray<nb::numpy, T, nb::ndim<1>> (*)(nb::ndarray<T, nb::c_contig>)>(&cumprod),
            "Cumulative product");
      
      // linear algebra operations
      m.def("matmul", static_cast<nb::ndarray<nb::numpy, T, nb::ndim<2>> (*)(nb::ndarray<T, nb::c_contig, nb::ndim<2>>, nb::ndarray<T, nb::c_contig, nb::ndim<2>>)>(&matmul),
      "Matrix multiplication using BLAS");
      m.def("trace", static_cast<T (*)(nb::ndarray<T, nb::c_contig, nb::ndim<2>>)>(&trace),
      "Matrix trace (sum of diagonal elements)");
      m.def("norm", static_cast<T (*)(nb::ndarray<T, nb::c_contig>)>(&norm),
      "Vector norm (Euclidean/L2)");
      m.def("dot", static_cast<T (*)(nb::ndarray<T, nb::c_contig>, nb::ndarray<T, nb::c_contig>)>(&dot),
      "Dot product of two vectors");

      // stats ops
      m.def("mean", static_cast<T (*)(nb::ndarray<T, nb::c_contig>)>(&mean),
            "Mean value");
      m.def("median", static_cast<T (*)(nb::ndarray<T, nb::c_contig>)>(&median),
            "Median value");
      m.def("mode", static_cast<T (*)(nb::ndarray<T, nb::c_contig>)>(&mode),
            "Mode value");
      m.def("variance", static_cast<T (*)(nb::ndarray<T, nb::c_contig>)>(&variance),
            "Variance");
      m.def("stddev", static_cast<T (*)(nb::ndarray<T, nb::c_contig>)>(&stddev),
            "Standard deviation");
            m.def("covariance", static_cast<double (*)(nb::ndarray<T, nb::c_contig>, nb::ndarray<T, nb::c_contig>)>(&covariance),
            "Covariance");
      m.def("covMatrix", static_cast<void (*)(nb::ndarray<double, nb::c_contig>, nb::ndarray<T, nb::c_contig>)>(&covMatrix),
            "Covariance matrix for a single array");
      m.def("covMatrix", static_cast<void (*)(nb::ndarray<double, nb::c_contig>, nb::ndarray<T, nb::c_contig>, 
                                             nb::ndarray<T, nb::c_contig>)>(&covMatrix),
            "Covariance matrix for two arrays");
      m.def("covMatrix", static_cast<void (*)(nb::ndarray<double, nb::c_contig>, nb::ndarray<T, nb::c_contig>, 
                                             nb::ndarray<T, nb::c_contig>, nb::ndarray<T, nb::c_contig>)>(&covMatrix),
            "Covariance matrix for three arrays");
      m.def("covMatrix", static_cast<void (*)(nb::ndarray<double, nb::c_contig>, nb::ndarray<T, nb::c_contig>, 
                                             nb::ndarray<T, nb::c_contig>, nb::ndarray<T, nb::c_contig>, 
                                             nb::ndarray<T, nb::c_contig>)>(&covMatrix),
            "Covariance matrix for four arrays");
      m.def("corrMatrix", static_cast<void (*)(nb::ndarray<double, nb::c_contig>, nb::ndarray<T, nb::c_contig>)>(&corrMatrix),
            "Correlation matrix for a single array");
      m.def("corrMatrix", static_cast<void (*)(nb::ndarray<double, nb::c_contig>, nb::ndarray<T, nb::c_contig>,
                                             nb::ndarray<T, nb::c_contig>)>(&corrMatrix),
            "Correlation matrix for two arrays");
      m.def("corrMatrix", static_cast<void (*)(nb::ndarray<double, nb::c_contig>, nb::ndarray<T, nb::c_contig>,
                                             nb::ndarray<T, nb::c_contig>, nb::ndarray<T, nb::c_contig>)>(&corrMatrix),
            "Correlation matrix for three arrays");
      m.def("corrMatrix", static_cast<void (*)(nb::ndarray<double, nb::c_contig>, nb::ndarray<T, nb::c_contig>,
                                             nb::ndarray<T, nb::c_contig>, nb::ndarray<T, nb::c_contig>,
                                             nb::ndarray<T, nb::c_contig>)>(&corrMatrix),
            "Correlation matrix for four arrays");
      m.def("correlation", static_cast<double (*)(nb::ndarray<T, nb::c_contig>, nb::ndarray<T, nb::c_contig>)>(&correlation),
            "Correlation");
      m.def("histogram", static_cast<void (*)(nb::ndarray<T, nb::c_contig>, nb::ndarray<T, nb::c_contig>, nb::ndarray<size_t, nb::c_contig>)>(&histogram),
            "Histogram");
}

} // registry