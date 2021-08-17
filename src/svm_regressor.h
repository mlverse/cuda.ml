#pragma once

#include <Rcpp.h>

#ifdef HAS_CUML

namespace cuml4r {

SEXP svr_fit(Rcpp::NumericMatrix const& X, Rcpp::NumericVector const& y,
             double const cost, int const kernel, double const gamma,
             double const coef0, int const degree, double const tol,
             int const max_iter, int const nochange_steps,
             double const cache_size, double epsilon,
             Rcpp::NumericVector const& sample_weights, int const verbosity);

Rcpp::NumericVector svr_predict(SEXP svr_xptr, Rcpp::NumericMatrix const& X);

}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
