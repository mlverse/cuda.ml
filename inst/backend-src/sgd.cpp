#include "sgd.h"

// [[Rcpp::export(".sgd_fit")]]
Rcpp::List sgd_fit(Rcpp::NumericMatrix const& x, Rcpp::NumericVector const& y,
                   bool const fit_intercept, int const batch_size,
                   int const epochs, int const lr_type, double const eta0,
                   double const power_t, int const loss, int const penalty,
                   double const alpha, double const l1_ratio,
                   bool const shuffle, double const tol,
                   int const n_iter_no_change) {
  return cuml4r::sgd_fit(x, y, fit_intercept, batch_size, epochs, lr_type, eta0,
                         power_t, loss, penalty, alpha, l1_ratio, shuffle, tol,
                         n_iter_no_change);
}
