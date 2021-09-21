#pragma once

#include <Rcpp.h>

#ifdef HAS_CUML

namespace cuml4r {

SEXP rf_classifier_fit(Rcpp::NumericMatrix const& input,
                       Rcpp::IntegerVector const& labels, int const n_trees,
                       bool const bootstrap, float const max_samples,
                       int const n_streams, int const max_depth,
                       int const max_leaves, float const max_features,
                       int const n_bins, int const min_samples_leaf,
                       int const min_samples_split, int const split_criterion,
                       float const min_impurity_decrease,
                       int const max_batch_size, int const verbosity);

Rcpp::IntegerVector rf_classifier_predict(SEXP model_xptr,
                                          Rcpp::NumericMatrix const& input,
                                          int const verbosity);

Rcpp::NumericMatrix rf_classifier_predict_class_probabilities(
  SEXP model_xptr, Rcpp::NumericMatrix const& input);

Rcpp::List rf_classifier_get_state(SEXP model);

SEXP rf_classifier_set_state(Rcpp::List const& state);

}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
