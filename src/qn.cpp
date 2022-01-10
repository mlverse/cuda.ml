#include "qn.h"

#ifndef HAS_CUML

#include "qn_constants.h"

namespace {

Rcpp::List cuml4r_example_qn_model() {
  // return a trivial model
  Rcpp::List model;
  model[cuml4r::qn::kCoefs] = Rcpp::NumericMatrix(3, 3);
  model[cuml4r::qn::kFitIntercept] = true;
  model[cuml4r::qn::kLossType] = 0;
  model[cuml4r::qn::kNumClasses] = 3;
  model[cuml4r::qn::kObjective] = 0;

  return model;
}

}  // namespace

#endif

// [[Rcpp::export(".qn_fit")]]
Rcpp::List qn_fit(Rcpp::NumericMatrix const& X, Rcpp::IntegerVector const& y,
                  int const n_classes, int const loss_type,
                  bool const fit_intercept, double const l1, double const l2,
                  int const max_iters, double const tol, double const delta,
                  int const linesearch_max_iters, int const lbfgs_memory,
                  Rcpp::NumericVector const& sample_weight) {
#ifdef HAS_CUML

  return cuml4r::qn_fit(X, y, n_classes, loss_type, fit_intercept, l1, l2,
                        max_iters, tol, delta, linesearch_max_iters,
                        lbfgs_memory, sample_weight);

#else

#include "warn_cuml_missing.h"

  return cuml4r_example_qn_model();

#endif
}

// [[Rcpp::export(".qn_predict")]]
Rcpp::NumericVector qn_predict(Rcpp::NumericMatrix const& X,
                               int const n_classes,
                               Rcpp::NumericMatrix const& coefs,
                               int const loss_type, bool const fit_intercept) {
#ifdef HAS_CUML

  return cuml4r::qn_predict(X, n_classes, coefs, loss_type, fit_intercept);

#else

#include "warn_cuml_missing.h"

  // return some dummy values
  return Rcpp::NumericVector(32);

#endif
}
