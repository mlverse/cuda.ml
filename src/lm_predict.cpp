#include "lm_predict.h"

// [[Rcpp::export(".lm_predict")]]
Rcpp::NumericVector lm_predict(SEXP input, SEXP coef, double const intercept) {
#ifdef HAS_CUML

  return cuml4r::lm_predict(input, coef, intercept);

#else

#include "warn_cuml_missing.h"

  // return some dummy values
  return Rcpp::NumericVector(32);

#endif
}
