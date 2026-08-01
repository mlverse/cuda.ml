#pragma once

#include <Rcpp.h>

namespace cuml4r {

Rcpp::List dbscan(Rcpp::NumericMatrix const& x, int const min_pts,
                  double const eps, size_t const max_bytes_per_batch,
                  int const verbosity);

}  // namespace cuml4r
