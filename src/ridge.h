#pragma once

#include <Rcpp.h>

#ifdef HAS_CUML

namespace cuml4r {

Rcpp::List ridge_fit(Rcpp::NumericMatrix const& x, Rcpp::NumericVector const& y,
                     bool const fit_intercept, bool const normalize_input,
                     double const alpha, int const algo);

}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
