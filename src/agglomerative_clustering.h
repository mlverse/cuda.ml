#pragma once

#include <Rcpp.h>

namespace cuml4r {

Rcpp::List agglomerative_clustering(Rcpp::NumericMatrix const& x,
                                    bool const pairwise_conn, int const metric,
                                    int const n_neighbors,
                                    int const n_clusters);

}  // namespace cuml4r
