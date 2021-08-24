#include "knn.h"

// [[Rcpp::export(".knn_fit")]]
SEXP knn_fit(Rcpp::NumericMatrix const& x, int const n_neighbors,
             int const algo, int const metric, float const p,
             Rcpp::List const& algo_params) {
#ifdef HAS_CUML

  return cuml4r::knn_fit(x, n_neighbors, algo, metric, p, algo_params);

#else

#include "warn_cuml_missing.h"

  return nullptr;

#endif
}
