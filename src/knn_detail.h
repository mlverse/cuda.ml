#pragma once

#include <Rcpp.h>

namespace cuml4r {
namespace knn {
namespace detail {

constexpr char kResponses[] = "responses";

template <typename T>
struct RcppVector {};

template <>
struct RcppVector<int> {
  using type = Rcpp::IntegerVector;
};

template <>
struct RcppVector<float> {
  using type = Rcpp::NumericVector;
};

}  // namespace detail
}  // namespace knn
}  // namespace cuml4r
