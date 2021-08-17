#pragma once

#include <Rcpp.h>
#include <treelite/c_api.h>

#ifndef CUML4R_TREELITE_C_API_MISSING

namespace cuml4r {

SEXP treelite_load_xgboost_model(
  std::string const& filename, int const algo, bool const output_class,
  float const threshold, int const storage_type, int const block_per_sm,
  int const threads_per_tree, int const n_items);

Rcpp::NumericMatrix treelite_predict(
    SEXP const& model,
    Rcpp::NumericMatrix const& x,
    bool const output_probabilities);

}  // namespace cuml4r

#endif
