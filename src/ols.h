#pragma once

#include <Rcpp.h>

namespace cuml4r {

Rcpp::List ols_fit(Rcpp::NumericMatrix const& x, Rcpp::NumericVector const& y,
                   bool const fit_intercept, int const algo);

}  // namespace cuml4r
