#pragma once

#include <Rcpp.h>

namespace cuml4r {

SEXP rf_regressor_fit(Rcpp::NumericMatrix const& input,
                      Rcpp::NumericVector const& responses, int n_trees,
                      bool bootstrap, float max_samples, int n_streams,
                      int max_depth, int max_leaves, float max_features,
                      int n_bins, int min_samples_leaf, int min_samples_split,
                      int split_criterion, float min_impurity_decrease,
                      int max_batch_size, int seed);

}  // namespace cuml4r
