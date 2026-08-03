#include "nvforest.h"

// [[Rcpp::export(".nvforest_load_model")]]
SEXP nvforest_load_model(std::string const& filename, int const model_type,
                         int const device, int const device_id,
                         int const layout, int const precision,
                         int const default_chunk_size, int const align_bytes) {
  return cuml4r::nvforest_load_model(filename, model_type, device, device_id,
                                     layout, precision, default_chunk_size,
                                     align_bytes);
}

// [[Rcpp::export(".nvforest_model_info")]]
Rcpp::List nvforest_model_info(SEXP model) {
  return cuml4r::nvforest_model_info(model);
}

// [[Rcpp::export(".nvforest_predict")]]
SEXP nvforest_predict(SEXP model, Rcpp::NumericMatrix const& input,
                      int const prediction_type, double const threshold,
                      int const chunk_size) {
  return cuml4r::nvforest_predict(model, input, prediction_type, threshold,
                                  chunk_size);
}

// [[Rcpp::export(".nvforest_serialize")]]
Rcpp::RawVector nvforest_serialize(SEXP model) {
  return cuml4r::nvforest_serialize(model);
}

// [[Rcpp::export(".nvforest_unserialize")]]
SEXP nvforest_unserialize(Rcpp::RawVector const& bytes, int const device,
                          int const device_id, int const layout,
                          int const precision, int const default_chunk_size,
                          int const align_bytes,
                          bool const averaged_vector_leaf_probabilities) {
  return cuml4r::nvforest_unserialize(
    bytes, device, device_id, layout, precision, default_chunk_size,
    align_bytes, averaged_vector_leaf_probabilities);
}
