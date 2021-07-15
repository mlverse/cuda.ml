#pragma once

#if HAS_CUML

#include <Rcpp.h>
#include "pinned_host_vector.h"

namespace cuml4r {

template <typename T = double>
struct Matrix {
  size_t const numRows;
  size_t const numCols;
  // all entries of the matrix in row-major order
  thrust::host_vector<T, thrust::cuda::experimental::pinned_allocator<T>> const
    values;

  explicit Matrix(Rcpp::NumericMatrix const& m, bool const transpose) noexcept
    : numRows(transpose ? m.ncol() : m.nrow()),
      numCols(transpose ? m.nrow() : m.ncol()),
      // conversion from column-major order to row-major order
      values(transpose ? Rcpp::as<std::vector<T>>(Rcpp::NumericVector(m))
                       : Rcpp::as<std::vector<T>>(
                           Rcpp::NumericVector(Rcpp::transpose(m)))) {}
};

}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
