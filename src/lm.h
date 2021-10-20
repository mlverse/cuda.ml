#pragma once

#include <Rcpp.h>
#include "lm_params.h"

#include <functional>

namespace raft {

class handle_t;

}  // namespace raft

namespace cuml4r {

// generic template for fitting linear models
Rcpp::List lm_fit(
  Rcpp::NumericMatrix const& x, Rcpp::NumericVector const& y,
  bool const fit_intercept, bool const normalize_input,
  std::function<void(raft::handle_t&, lm::Params const&)> const& fit_impl);

}  // namespace cuml4r
