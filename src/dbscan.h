#pragma once

#include <Rcpp.h>

#ifdef HAS_CUML

namespace cuml4r {

Rcpp::List dbscan(Rcpp::NumericMatrix const& x, int const min_pts,
                  double const eps, size_t const max_bytes_per_batch,
                  int const verbosity);

}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
