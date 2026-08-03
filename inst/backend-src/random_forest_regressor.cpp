#include "random_forest_regressor.h"

// [[Rcpp::export(".rf_regressor_fit")]]
SEXP rf_regressor_fit(Rcpp::NumericMatrix const& input,
                      Rcpp::NumericVector const& responses, int const n_trees,
                      bool const bootstrap, float const max_samples,
                      int const n_streams, int const max_depth,
                      int const max_leaves, float const max_features,
                      int const n_bins, int const min_samples_leaf,
                      int const min_samples_split, int const split_criterion,
                      float const min_impurity_decrease,
                      int const max_batch_size, int const seed) {
  return cuml4r::rf_regressor_fit(
    input, responses, n_trees, bootstrap, max_samples, n_streams, max_depth,
    max_leaves, max_features, n_bins, min_samples_leaf, min_samples_split,
    split_criterion, min_impurity_decrease, max_batch_size, seed);
}
