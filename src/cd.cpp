#include "cd.h"

#ifndef HAS_CUML

#include "example_linear_model.h"

#endif

// [[Rcpp::export(".cd_fit")]]
Rcpp::List cd_fit(Rcpp::NumericMatrix const& x, Rcpp::NumericVector const& y,
                  bool const fit_intercept, bool const normalize_input,
                  int const epochs, int const loss, double const alpha,
                  double const l1_ratio, bool const shuffle, double const tol) {
#ifdef HAS_CUML

  return cuml4r::cd_fit(x, y, fit_intercept, normalize_input, epochs, loss,
                        alpha, l1_ratio, shuffle, tol);

#else

#include "warn_cuml_missing.h"

  return cuml4r_example_linear_model();

#endif
}
