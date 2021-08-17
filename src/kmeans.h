#pragma once

#include <Rcpp.h>

#ifdef HAS_CUML

namespace cuml4r {

Rcpp::List kmeans(Rcpp::NumericMatrix const& x, int const k,
                  int const max_iters, double const tol, int const init_method,
                  Rcpp::NumericMatrix const& centroids, int const seed,
                  int const verbosity);

}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
