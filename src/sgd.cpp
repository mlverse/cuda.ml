#include "sgd.h"

#ifndef HAS_CUML

#include "example_linear_model.h"

#endif

// [[Rcpp::export(".sgd_fit")]]
Rcpp::List sgd_fit(Rcpp::NumericMatrix const& x, Rcpp::NumericVector const& y,
                   bool const fit_intercept, int const batch_size,
                   int const epochs, int const lr_type, double const eta0,
                   double const power_t, int const loss, int const penalty,
                   double const alpha, double const l1_ratio,
                   bool const shuffle, double const tol,
                   int const n_iter_no_change) {
#ifdef HAS_CUML

  return cuml4r::sgd_fit(x, y, fit_intercept, batch_size, epochs, lr_type, eta0,
                         power_t, loss, penalty, alpha, l1_ratio, shuffle, tol,
                         n_iter_no_change);

#else

#include "warn_cuml_missing.h"

  return cuml4r_example_linear_model();

#endif
}
