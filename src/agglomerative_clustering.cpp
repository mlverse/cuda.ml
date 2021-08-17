#include "agglomerative_clustering.h"

// [[Rcpp::export(".agglomerative_clustering")]]
Rcpp::List agglomerative_clustering(Rcpp::NumericMatrix const& x,
                                    bool const pairwise_conn, int const metric,
                                    int const n_neighbors,
                                    int const n_clusters) {
#ifdef HAS_CUML

  return cuml4r::agglomerative_clustering(x, pairwise_conn, metric, n_neighbors,
                                          n_clusters);

#else

#include "warn_cuml_missing.h"

  return {};

#endif
}
