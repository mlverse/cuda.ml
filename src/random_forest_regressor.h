#pragma once

#include <Rcpp.h>

#ifdef HAS_CUML

namespace cuml4r {

SEXP rf_regressor_fit(Rcpp::NumericMatrix const& input,
                      Rcpp::NumericVector const& responses, int const n_trees,
                      bool const bootstrap, float const max_samples,
                      int const n_streams, int const max_depth,
                      int const max_leaves, float const max_features,
                      int const n_bins, int const min_samples_leaf,
                      int const min_samples_split, int const split_criterion,
                      float const min_impurity_decrease,
                      int const max_batch_size, int const verbosity);

Rcpp::NumericVector rf_regressor_predict(SEXP model_xptr,
                                         Rcpp::NumericMatrix const& input,
                                         int const verbosity);

Rcpp::List rf_regressor_get_state(SEXP model);

SEXP rf_regressor_set_state(Rcpp::List const& state);

}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
