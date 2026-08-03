#include "kmeans.h"

// [[Rcpp::export(".kmeans")]]
Rcpp::List kmeans(Rcpp::NumericMatrix const& x, int const k,
                  int const max_iters, double const tol, int const init_method,
                  Rcpp::NumericMatrix const& centroids, int const seed,
                  int const verbosity) {
  return cuml4r::kmeans(x, k, max_iters, tol, init_method, centroids, seed,
                        verbosity);
}
