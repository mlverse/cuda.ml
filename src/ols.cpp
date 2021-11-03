#include "ols.h"

#ifndef HAS_CUML

#include "example_linear_model.h"

#endif

// [[Rcpp::export(".ols_fit")]]
Rcpp::List ols_fit(Rcpp::NumericMatrix const& x, Rcpp::NumericVector const& y,
                   bool const fit_intercept, bool const normalize_input,
                   int const algo) {
#ifdef HAS_CUML

  return cuml4r::ols_fit(x, y, fit_intercept, normalize_input, algo);

#else

#include "warn_cuml_missing.h"

  return cuml4r_example_linear_model();

#endif
}
