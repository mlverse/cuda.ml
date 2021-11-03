#pragma once

#include <Rcpp.h>
#include "lm_constants.h"
#include "lm_params.h"

#include <functional>

#ifdef HAS_CUML

namespace raft {

class handle_t;

}  // namespace raft

#else

#include "warn_cuml_missing.h"

#endif

namespace cuml4r {

#ifdef HAS_CUML

// generic template for fitting linear models
Rcpp::List lm_fit(
  Rcpp::NumericMatrix const& x, Rcpp::NumericVector const& y,
  lm::InterceptType const intercept_type, bool const fit_intercept,
  bool const normalize_input,
  std::function<void(raft::handle_t&, lm::Params const&)> const& fit_impl);

#else

#include "warn_cuml_missing.h"

#endif

}  // namespace cuml4r
