#include "fil.h"

// [[Rcpp::export(".fil_enabled")]]
bool fil_enabled() {
#ifdef CUML4R_TREELITE_C_API_MISSING

  return false;

#else

  return true;

#endif
}

// [[Rcpp::export(".fil_load_model")]]
SEXP fil_load_model(int const model_type, std::string const& filename,
                    int const algo, bool const output_class,
                    float const threshold, int const storage_type,
                    int const blocks_per_sm, int const threads_per_tree,
                    int const n_items) {
#ifndef CUML4R_TREELITE_C_API_MISSING

  return cuml4r::fil_load_model(model_type, filename, algo, output_class,
                                threshold, storage_type, blocks_per_sm,
                                threads_per_tree, n_items);

#else

  return nullptr;

#endif
}

// [[Rcpp::export(".fil_predict")]]
Rcpp::NumericMatrix fil_predict(SEXP const& model, Rcpp::NumericMatrix const& x,
                                bool const output_probabilities) {
#ifndef CUML4R_TREELITE_C_API_MISSING

  return cuml4r::fil_predict(model, x, output_probabilities);

#else

  return {};

#endif
}
