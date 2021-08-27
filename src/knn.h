#pragma once

#include <Rcpp.h>

#ifdef HAS_CUML

namespace cuml4r {

SEXP knn_fit(Rcpp::NumericMatrix const& x, Rcpp::IntegerVector const& y,
             int const algo, int const metric, float const p,
             Rcpp::List const& algo_params);

Rcpp::IntegerVector knn_classifier_predict(Rcpp::List const& model,
                                           Rcpp::NumericMatrix const& x,
                                           int const n_neighbors);

Rcpp::NumericMatrix knn_classifier_predict_probabilities(
  Rcpp::List const& model, Rcpp::NumericMatrix const& x, int const n_neighbors);

}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
