#include "qn.h"

// [[Rcpp::export(".qn_fit")]]
Rcpp::List qn_fit(Rcpp::NumericMatrix const& X, Rcpp::IntegerVector const& y,
                  int const n_classes, int const loss_type,
                  bool const fit_intercept, double const l1, double const l2,
                  int const max_iters, double const tol, double const delta,
                  int const linesearch_max_iters, int const lbfgs_memory,
                  bool const penalty_normalized,
                  Rcpp::NumericVector const& sample_weight) {
  return cuml4r::qn_fit(X, y, n_classes, loss_type, fit_intercept, l1, l2,
                        max_iters, tol, delta, linesearch_max_iters,
                        lbfgs_memory, penalty_normalized, sample_weight);
}

// [[Rcpp::export(".qn_predict_probabilities")]]
Rcpp::NumericMatrix qn_predict_probabilities(Rcpp::NumericMatrix const& X,
                                             int const n_classes,
                                             Rcpp::NumericMatrix const& coefs,
                                             int const loss_type,
                                             bool const fit_intercept) {
  return cuml4r::qn_predict_probabilities(X, n_classes, coefs, loss_type,
                                          fit_intercept);
}

// [[Rcpp::export(".qn_predict")]]
Rcpp::NumericVector qn_predict(Rcpp::NumericMatrix const& X,
                               int const n_classes,
                               Rcpp::NumericMatrix const& coefs,
                               int const loss_type, bool const fit_intercept) {
  return cuml4r::qn_predict(X, n_classes, coefs, loss_type, fit_intercept);
}
