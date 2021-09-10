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
                    int const algo, bool const classification,
                    float const threshold, int const storage_type,
                    int const blocks_per_sm, int const threads_per_tree,
                    int const n_items) {
#ifndef CUML4R_TREELITE_C_API_MISSING

  return cuml4r::fil_load_model(model_type, filename, algo, classification,
                                threshold, storage_type, blocks_per_sm,
                                threads_per_tree, n_items);

#else

  return Rcpp::List();

#endif
}

// [[Rcpp::export(".fil_get_num_classes")]]
int fil_get_num_classes(SEXP const& model) {
#ifndef CUML4R_TREELITE_C_API_MISSING

  return cuml4r::fil_get_num_classes(model);

#else

  return -1;

#endif
}

// [[Rcpp::export(".fil_predict")]]
Rcpp::NumericMatrix fil_predict(SEXP const& model, Rcpp::NumericMatrix const& x,
                                bool const output_class_probabilities) {
#ifndef CUML4R_TREELITE_C_API_MISSING

  return cuml4r::fil_predict(model, x, output_class_probabilities);

#else

  Rcpp::NumericVector const preds(x.nrow(), 1.0);
  return Rcpp::NumericMatrix(x.nrow(), 1, preds.cbegin());

#endif
}
