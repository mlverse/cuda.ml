#pragma once

#include <Rcpp.h>

#ifdef HAS_CUML

namespace cuml4r {

Rcpp::NumericVector lm_predict(Rcpp::NumericMatrix const& input,
                               Rcpp::NumericVector const& coef,
                               double const intercept);

}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
