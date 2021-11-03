#include "ridge.h"

#ifndef HAS_CUML

#include "example_linear_model.h"

#endif

// [[Rcpp::export(".ridge_fit")]]
Rcpp::List ridge_fit(Rcpp::NumericMatrix const& x, Rcpp::NumericVector const& y,
                     bool const fit_intercept, bool const normalize_input,
                     double const alpha, int const algo) {
#ifdef HAS_CUML

  return cuml4r::ridge_fit(x, y, fit_intercept, normalize_input, alpha, algo);

#else

#include "warn_cuml_missing.h"

  return cuml4r_example_linear_model();

#endif
}
