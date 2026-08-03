#pragma once

#include <Rcpp.h>

namespace cuml4r {

Rcpp::List umap_fit(Rcpp::NumericMatrix const& x, Rcpp::NumericVector const& y,
                    int const n_neighbors, int const n_components,
                    int const n_epochs, float const learning_rate,
                    float const min_dist, float const spread,
                    float const set_op_mix_ratio, int const local_connectivity,
                    float const repulsion_strength,
                    int const negative_sample_rate,
                    float const transform_queue_size, int const verbosity,
                    float const a, float const b, int const init,
                    int const target_n_neighbors, int const target_metric,
                    float const target_weight, uint64_t const random_state,
                    bool const deterministic);

Rcpp::NumericMatrix umap_transform(Rcpp::List const& model,
                                   Rcpp::NumericMatrix const& x);

Rcpp::List umap_get_state(Rcpp::List const& model);
Rcpp::List umap_set_state(Rcpp::List const& state);

}  // namespace cuml4r
