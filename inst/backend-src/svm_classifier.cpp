#include "svm_classifier.h"

// [[Rcpp::export(".svc_fit")]]
SEXP svc_fit(Rcpp::NumericMatrix const& input,
             Rcpp::NumericVector const& labels, double const cost,
             int const kernel, double const gamma, double const coef0,
             int const degree, double const tol, int const max_iter,
             int const nochange_steps, double const cache_size,
             Rcpp::NumericVector const& sample_weights, int const verbosity) {
  return cuml4r::svc_fit(input, labels, cost, kernel, gamma, coef0, degree, tol,
                         max_iter, nochange_steps, cache_size, sample_weights,
                         verbosity);
}

// [[Rcpp::export(".svc_predict")]]
SEXP svc_predict(SEXP model_xptr, Rcpp::NumericMatrix const& input,
                 bool predict_class) {
  return cuml4r::svc_predict(model_xptr, input, predict_class);
}

// [[Rcpp::export(".svc_get_state")]]
Rcpp::List svc_get_state(SEXP model) { return cuml4r::svc_get_state(model); }

// [[Rcpp::export(".svc_set_state")]]
SEXP svc_set_state(Rcpp::List const& state) {
  return cuml4r::svc_set_state(state);
}
