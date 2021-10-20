#pragma once

#include <Rcpp.h>

#ifdef HAS_CUML

namespace cuml4r {

Rcpp::List ols_fit(Rcpp::NumericMatrix const& x, Rcpp::NumericVector const& y,
                   bool const fit_intercept, bool const normalize_input,
                   int const algo);

}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
