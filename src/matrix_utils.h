#pragma once

#ifdef HAS_CUML

#include <Rcpp.h>
#include "pinned_host_vector.h"

namespace cuml4r {

template <typename T = double>
struct Matrix {
  size_t const numRows;
  size_t const numCols;
  // all entries of the matrix in row-major order
  pinned_host_vector<T> const values;

  explicit Matrix(Rcpp::NumericMatrix const& m, bool const transpose) noexcept
    : numRows(transpose ? m.ncol() : m.nrow()),
      numCols(transpose ? m.nrow() : m.ncol()),
      // conversion from column-major order to row-major order
      values(
        Rcpp::as<pinned_host_vector<T>>(transpose ? m : Rcpp::transpose(m))) {}
};

}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
