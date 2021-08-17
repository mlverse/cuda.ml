#include "svm_regressor.h"

// [[Rcpp::export(".svr_fit")]]
SEXP svr_fit(Rcpp::NumericMatrix const& X, Rcpp::NumericVector const& y,
             double const cost, int const kernel, double const gamma,
             double const coef0, int const degree, double const tol,
             int const max_iter, int const nochange_steps,
             double const cache_size, double epsilon,
             Rcpp::NumericVector const& sample_weights, int const verbosity) {
#ifdef HAS_CUML

  return cuml4r::svr_fit(X, y, cost, kernel, gamma, coef0, degree, tol,
                         max_iter, nochange_steps, cache_size, epsilon,
                         sample_weights, verbosity);

#else

#include "warn_cuml_missing.h"

  return Rcpp::List();

#endif
}

// [[Rcpp::export(".svr_predict")]]
Rcpp::NumericVector svr_predict(SEXP svr_xptr, Rcpp::NumericMatrix const& X) {
#ifdef HAS_CUML

  return cuml4r::svr_predict(svr_xptr, X);

#else

#include "warn_cuml_missing.h"

  return {};

#endif
}
