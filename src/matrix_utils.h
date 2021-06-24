#pragma once

namespace cuml4r {

template <typename T = double>
struct Matrix {
   size_t const numRows;
   size_t const numCols; 
   // all entries of the matrix in row-major order
   std::vector<T> const values;

   explicit Matrix(Rcpp::NumericMatrix const& m, bool const transpose) noexcept :
     numRows(transpose ? m.ncol() : m.nrow()),
     numCols(transpose ? m.nrow() : m.ncol()),
     // conversion from column-major order to row-major order
     values(
       transpose ? 
       Rcpp::as<std::vector<T>>(Rcpp::NumericVector(m)) :
       Rcpp::as<std::vector<T>>(Rcpp::NumericVector(Rcpp::transpose(m)))
     ) {}
};

}
