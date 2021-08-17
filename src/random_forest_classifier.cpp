#include "random_forest_classifier.h"

// [[Rcpp::export(".rf_classifier_fit")]]
SEXP rf_classifier_fit(Rcpp::NumericMatrix const& input,
                       Rcpp::IntegerVector const& labels, int const n_trees,
                       bool const bootstrap, float const max_samples,
                       int const n_streams, int const max_depth,
                       int const max_leaves, float const max_features,
                       int const n_bins, int const min_samples_leaf,
                       int const min_samples_split, int const split_criterion,
                       float const min_impurity_decrease,
                       int const max_batch_size, int const verbosity) {
#ifdef HAS_CUML

  return cuml4r::rf_classifier_fit(
    input, labels, n_trees, bootstrap, max_samples, n_streams, max_depth,
    max_leaves, max_features, n_bins, min_samples_leaf, min_samples_split,
    split_criterion, min_impurity_decrease, max_batch_size, verbosity);
#else

#include "warn_cuml_missing.h"

  return Rcpp::List();

#endif
}

// [[Rcpp::export(".rf_classifier_predict")]]
Rcpp::IntegerVector rf_classifier_predict(SEXP model_xptr,
                                          Rcpp::NumericMatrix const& input,
                                          int const verbosity) {
#ifdef HAS_CUML

  return cuml4r::rf_classifier_predict(model_xptr, input, verbosity);

#else

#include "warn_cuml_missing.h"

  return {};

#endif
}
