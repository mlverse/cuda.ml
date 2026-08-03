#include "ols.h"

// [[Rcpp::export(".ols_fit")]]
Rcpp::List ols_fit(Rcpp::NumericMatrix const& x, Rcpp::NumericVector const& y,
                   bool const fit_intercept, int const algo) {
  return cuml4r::ols_fit(x, y, fit_intercept, algo);
}
