#include "knn.h"

// [[Rcpp::export(".knn_classifier_fit")]]
SEXP knn_classifier_fit(Rcpp::NumericMatrix const& x,
                        Rcpp::IntegerVector const& y, int const algo,
                        int const metric, float const p,
                        Rcpp::List const& algo_params) {
#ifdef HAS_CUML

  return cuml4r::knn_fit(x, y, algo, metric, p, algo_params);

#else

#include "warn_cuml_missing.h"

  return nullptr;

#endif
}

// [[Rcpp::export(".knn_classifier_predict")]]
Rcpp::IntegerVector knn_classifier_predict(Rcpp::List const& model,
                                           Rcpp::NumericMatrix const& x,
                                           int const n_neighbors) {
#ifdef HAS_CUML

  return cuml4r::knn_classifier_predict(model, x, n_neighbors);
#else

#include "warn_cuml_missing.h"

  return {};

#endif
}

// [[Rcpp::export(".knn_classifier_predict_probabilities")]]
Rcpp::NumericMatrix knn_classifier_predict_probabilities(
  Rcpp::List const& model, Rcpp::NumericMatrix const& x,
  int const n_neighbors) {
#ifdef HAS_CUML

  return cuml4r::knn_classifier_predict_probabilities(model, x, n_neighbors);
#else

#include "warn_cuml_missing.h"

  return {};

#endif
}

// TODO:
// // [[Rcpp::export(".knn_regressor_fit")]]
// SEXP knn_regressor_fit(Rcpp::NumericMatrix const& x, Rcpp::NumericVector const& y,
//              int const algo, int const metric, float const p,
//              Rcpp::List const& algo_params) {
// #ifdef HAS_CUML
//
//   return cuml4r::knn_fit(x, y, algo, metric, p, algo_params);
//
// #else
//
// #include "warn_cuml_missing.h"
//
//   return nullptr;
//
// #endif
// }
