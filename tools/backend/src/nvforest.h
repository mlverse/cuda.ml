#pragma once

#include <Rcpp.h>

namespace cuml4r {

SEXP nvforest_load_model(std::string const& filename, int model_type,
                         int device, int device_id, int layout, int precision,
                         int default_chunk_size, int align_bytes);

Rcpp::List nvforest_model_info(SEXP model);

SEXP nvforest_predict(SEXP model, Rcpp::NumericMatrix const& input,
                      int prediction_type, double threshold, int chunk_size);

Rcpp::RawVector nvforest_serialize(SEXP model);

SEXP nvforest_unserialize(Rcpp::RawVector const& bytes, int device,
                          int device_id, int layout, int precision,
                          int default_chunk_size, int align_bytes,
                          bool averaged_vector_leaf_probabilities);

}  // namespace cuml4r
