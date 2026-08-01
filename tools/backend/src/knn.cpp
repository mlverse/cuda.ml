#include "knn.h"

// [[Rcpp::export(".knn_classifier_fit")]]
SEXP knn_classifier_fit(Rcpp::NumericMatrix const& x,
                        Rcpp::IntegerVector const& y, int const algo,
                        int const metric, float const p,
                        Rcpp::List const& algo_params) {
  return cuml4r::knn_fit(x, y, algo, metric, p, algo_params);
}

// [[Rcpp::export(".knn_classifier_predict")]]
Rcpp::IntegerVector knn_classifier_predict(Rcpp::List const& model,
                                           Rcpp::NumericMatrix const& x,
                                           int const n_neighbors) {
  return cuml4r::knn_classifier_predict(model, x, n_neighbors);
}

// [[Rcpp::export(".knn_classifier_predict_probabilities")]]
Rcpp::NumericMatrix knn_classifier_predict_probabilities(
  Rcpp::List const& model, Rcpp::NumericMatrix const& x,
  int const n_neighbors) {
  return cuml4r::knn_classifier_predict_probabilities(model, x, n_neighbors);
}

// [[Rcpp::export(".knn_regressor_fit")]]
SEXP knn_regressor_fit(Rcpp::NumericMatrix const& x,
                       Rcpp::NumericVector const& y, int const algo,
                       int const metric, float const p,
                       Rcpp::List const& algo_params) {
  return cuml4r::knn_fit(x, y, algo, metric, p, algo_params);
}

// [[Rcpp::export(".knn_regressor_predict")]]
Rcpp::NumericVector knn_regressor_predict(Rcpp::List const& model,
                                          Rcpp::NumericMatrix const& x,
                                          int const n_neighbors) {
  return cuml4r::knn_regressor_predict(model, x, n_neighbors);
}
