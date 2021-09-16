#include "umap.h"

// [[Rcpp::export(".umap_fit")]]
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
                    bool const deterministic) {
  Rcpp::List model;

#ifdef HAS_CUML

  return cuml4r::umap_fit(x, y, n_neighbors, n_components, n_epochs,
                          learning_rate, min_dist, spread, set_op_mix_ratio,
                          local_connectivity, repulsion_strength,
                          negative_sample_rate, transform_queue_size, verbosity,
                          a, b, init, target_n_neighbors, target_metric,
                          target_weight, random_state, deterministic);

#else

#include "warn_cuml_missing.h"

#endif

  return model;
}

// [[Rcpp::export(".umap_transform")]]
Rcpp::NumericMatrix umap_transform(Rcpp::List const& model,
                                   Rcpp::NumericMatrix const& x) {
#ifdef HAS_CUML

  return cuml4r::umap_transform(model, x);

#else

#include "warn_cuml_missing.h"

  // dummy values with distinct data points
  return Rcpp::NumericMatrix::diag(x.nrow(), 1);

#endif
}

// [[Rcpp::export(".umap_get_state")]]
Rcpp::List umap_get_state(Rcpp::List const& model) {
#ifdef HAS_CUML

  return cuml4r::umap_get_state(model);

#else

#include "warn_cuml_missing.h"

  return Rcpp::List();

#endif
}

// [[Rcpp::export(".umap_set_state")]]
Rcpp::List umap_set_state(Rcpp::List const& state) {
#ifdef HAS_CUML

  return cuml4r::umap_set_state(state);

#else

#include "warn_cuml_missing.h"

  return Rcpp::List();

#endif
}
