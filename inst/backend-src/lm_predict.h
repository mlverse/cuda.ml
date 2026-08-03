#pragma once

#include <Rcpp.h>

namespace cuml4r {

Rcpp::NumericVector lm_predict(Rcpp::NumericMatrix const& input,
                               Rcpp::NumericVector const& coef,
                               double const intercept);

}  // namespace cuml4r
