#include "svm_regressor.h"

// [[Rcpp::export(".svr_fit")]]
SEXP svr_fit(Rcpp::NumericMatrix const& X, Rcpp::NumericVector const& y,
             double const cost, int const kernel, double const gamma,
             double const coef0, int const degree, double const tol,
             int const max_iter, int const nochange_steps,
             double const cache_size, double epsilon,
             Rcpp::NumericVector const& sample_weights, int const verbosity) {
  return cuml4r::svr_fit(X, y, cost, kernel, gamma, coef0, degree, tol,
                         max_iter, nochange_steps, cache_size, epsilon,
                         sample_weights, verbosity);
}

// [[Rcpp::export(".svr_predict")]]
Rcpp::NumericVector svr_predict(SEXP svr_xptr, Rcpp::NumericMatrix const& X) {
  return cuml4r::svr_predict(svr_xptr, X);
}

// [[Rcpp::export(".svr_get_state")]]
Rcpp::List svr_get_state(SEXP model) { return cuml4r::svr_get_state(model); }

// [[Rcpp::export(".svr_set_state")]]
SEXP svr_set_state(Rcpp::List const& state) {
  return cuml4r::svr_set_state(state);
}
