#pragma once

#include <Rcpp.h>

namespace cuml4r {

Rcpp::List cd_fit(Rcpp::NumericMatrix const& x, Rcpp::NumericVector const& y,
                  bool const fit_intercept, int const epochs, int const loss,
                  double const alpha, double const l1_ratio, bool const shuffle,
                  double const tol);

}  // namespace cuml4r
