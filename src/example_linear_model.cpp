#include "example_linear_model.h"
#include "lm.h"

Rcpp::List cuml4r_example_linear_model() {
  // return a trivial model
  Rcpp::List model;
  model[cuml4r::lm::kCoef] = Rcpp::NumericVector(10);
  model[cuml4r::lm::kIntercept] = 0;

  return model;
}
