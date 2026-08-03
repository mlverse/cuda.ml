#pragma once

#include <Rcpp.h>

namespace cuml4r {

Rcpp::List ridge_fit(Rcpp::NumericMatrix const& x, Rcpp::NumericVector const& y,
                     bool const fit_intercept, double const alpha,
                     int const algo);

}  // namespace cuml4r
