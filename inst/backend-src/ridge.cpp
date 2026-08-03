#include "ridge.h"

// [[Rcpp::export(".ridge_fit")]]
Rcpp::List ridge_fit(Rcpp::NumericMatrix const& x, Rcpp::NumericVector const& y,
                     bool const fit_intercept, double const alpha,
                     int const algo) {
  return cuml4r::ridge_fit(x, y, fit_intercept, alpha, algo);
}
