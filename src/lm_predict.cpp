#include "lm_predict.h"

// [[Rcpp::export(".lm_predict")]]
Rcpp::NumericVector lm_predict(SEXP input, SEXP coef, double const intercept) {
  return cuml4r::lm_predict(input, coef, intercept);
}
