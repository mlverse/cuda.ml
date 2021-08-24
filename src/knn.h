#pragma once

#include <Rcpp.h>

#ifdef HAS_CUML

namespace cuml4r {

SEXP knn_fit(Rcpp::NumericMatrix const& x, int const n_neighbors,
             int const algo, int const metric, float const p,
             Rcpp::List const& algo_params);

}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
