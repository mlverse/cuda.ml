#include "dbscan.h"

// [[Rcpp::export(".dbscan")]]
Rcpp::List dbscan(Rcpp::NumericMatrix const& x, int const min_pts,
                  double const eps, size_t const max_bytes_per_batch,
                  int const verbosity) {
#ifdef HAS_CUML

  return cuml4r::dbscan(x, min_pts, eps, max_bytes_per_batch, verbosity);

#else

#include "warn_cuml_missing.h"

  return {};

#endif
}
