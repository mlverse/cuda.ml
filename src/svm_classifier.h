#pragma once

#include <Rcpp.h>

#ifdef HAS_CUML

namespace cuml4r {

SEXP svc_fit(Rcpp::NumericMatrix const& input,
             Rcpp::NumericVector const& labels, double const cost,
             int const kernel, double const gamma, double const coef0,
             int const degree, double const tol, int const max_iter,
             int const nochange_steps, double const cache_size,
             Rcpp::NumericVector const& sample_weights, int const verbosity);

SEXP svc_predict(SEXP model_xptr, Rcpp::NumericMatrix const& input,
                 bool predict_class);

}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
