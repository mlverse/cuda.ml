#pragma once

#include "knn_detail.h"

#ifdef HAS_CUML

namespace cuml4r {

Rcpp::List knn_fit(Rcpp::NumericMatrix const& x, int const algo,
                   int const metric, float const p,
                   Rcpp::List const& algo_params);

template <typename ResponseT>
Rcpp::List knn_fit(Rcpp::NumericMatrix const& x, ResponseT&& y, int const algo,
                   int const metric, float const p,
                   Rcpp::List const& algo_params) {
  auto model = knn_fit(x, algo, metric, p, algo_params);
  model[knn::detail::kResponses] = std::forward<ResponseT>(y);

  return model;
}

Rcpp::IntegerVector knn_classifier_predict(Rcpp::List const& model,
                                           Rcpp::NumericMatrix const& x,
                                           int const n_neighbors);

Rcpp::NumericMatrix knn_classifier_predict_probabilities(
  Rcpp::List const& model, Rcpp::NumericMatrix const& x, int const n_neighbors);

Rcpp::NumericVector knn_regressor_predict(Rcpp::List const& model,
                                          Rcpp::NumericMatrix const& x,
                                          int const n_neighbors);

}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
