#pragma once

#include <Rcpp.h>
#include <treelite/c_api.h>

#ifndef CUML4R_TREELITE_C_API_MISSING

namespace cuml4r {

SEXP fil_load_model(int const model_type, std::string const& filename,
                    int const algo, bool const classification,
                    float const threshold, int const storage_type,
                    int const block_per_sm, int const threads_per_tree,
                    int const n_items);

int fil_get_num_classes(SEXP const& model);

Rcpp::NumericMatrix fil_predict(SEXP const& model, Rcpp::NumericMatrix const& x,
                                bool const output_class_probabilities);

}  // namespace cuml4r

#endif
