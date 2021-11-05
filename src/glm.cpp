#include "glm.h"

// [[Rcpp::export(".glm_predict")]]
Rcpp::NumericVector glm_predict(SEXP input, SEXP coef, double const intercept) {
#ifdef HAS_CUML

  return cuml4r::glm_predict(input, coef, intercept);

#else

#include "warn_cuml_missing.h"

  // return some dummy values
  return Rcpp::NumericVector(32);

#endif
}
