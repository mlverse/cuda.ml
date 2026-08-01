#pragma once

#include <Rcpp.h>

namespace cuml4r {

Rcpp::List kmeans(Rcpp::NumericMatrix const& x, int const k,
                  int const max_iters, double const tol, int const init_method,
                  Rcpp::NumericMatrix const& centroids, int const seed,
                  int const verbosity);

}  // namespace cuml4r
