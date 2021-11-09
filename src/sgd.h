#pragma once

#include <Rcpp.h>

#ifdef HAS_CUML

namespace cuml4r {

Rcpp::List sgd_fit(Rcpp::NumericMatrix const& x, Rcpp::NumericVector const& y,
                   bool const fit_intercept, int const batch_size,
                   int const epochs, int const lr_type, double const eta0,
                   double const power_t, int const loss, int const penalty,
                   double const alpha, double const l1_ratio,
                   bool const shuffle, double const tol,
                   int const n_iter_no_change);

}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
