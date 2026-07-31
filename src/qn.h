#pragma once

#include <Rcpp.h>

namespace cuml4r {

Rcpp::List qn_fit(Rcpp::NumericMatrix const& X, Rcpp::IntegerVector const& y,
                  int const n_classes, int const loss_type,
                  bool const fit_intercept, double const l1, double const l2,
                  int const max_iters, double const tol, double const delta,
                  int const linesearch_max_iters, int const lbfgs_memory,
                  bool penalty_normalized,
                  Rcpp::NumericVector const& sample_weight);

Rcpp::NumericVector qn_predict(Rcpp::NumericMatrix const& X,
                               int const n_classes,
                               Rcpp::NumericMatrix const& coefs,
                               int const loss_type, bool const fit_intercept);

Rcpp::NumericMatrix qn_predict_probabilities(Rcpp::NumericMatrix const& X,
                                             int n_classes,
                                             Rcpp::NumericMatrix const& coefs,
                                             int loss_type, bool fit_intercept);

}  // namespace cuml4r
