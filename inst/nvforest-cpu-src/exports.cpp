#include <Rcpp.h>

Rcpp::List nvforest_backend_versions();
SEXP nvforest_load_model(std::string const &filename, int model_type,
                         int device, int device_id, int layout, int precision,
                         int default_chunk_size, int align_bytes);
Rcpp::List nvforest_model_info(SEXP model);
SEXP nvforest_predict(SEXP model, Rcpp::NumericMatrix const &input,
                      int prediction_type, double threshold, int chunk_size);
Rcpp::RawVector nvforest_serialize(SEXP model);
SEXP nvforest_unserialize(Rcpp::RawVector const &bytes, int device,
                          int device_id, int layout, int precision,
                          int default_chunk_size, int align_bytes,
                          bool averaged_vector_leaf_probabilities);

RcppExport SEXP _cuda_ml_nvforest_backend_versions() {
  BEGIN_RCPP
  return Rcpp::wrap(nvforest_backend_versions());
  END_RCPP
}

RcppExport SEXP _cuda_ml_nvforest_cpu_load_model(
    SEXP filename_sexp, SEXP model_type_sexp, SEXP device_sexp,
    SEXP device_id_sexp, SEXP layout_sexp, SEXP precision_sexp,
    SEXP chunk_size_sexp, SEXP align_bytes_sexp) {
  BEGIN_RCPP
  Rcpp::traits::input_parameter<std::string const &>::type filename(
      filename_sexp);
  return Rcpp::wrap(nvforest_load_model(
      filename, Rcpp::as<int>(model_type_sexp), Rcpp::as<int>(device_sexp),
      Rcpp::as<int>(device_id_sexp), Rcpp::as<int>(layout_sexp),
      Rcpp::as<int>(precision_sexp), Rcpp::as<int>(chunk_size_sexp),
      Rcpp::as<int>(align_bytes_sexp)));
  END_RCPP
}

RcppExport SEXP _cuda_ml_nvforest_cpu_model_info(SEXP model) {
  BEGIN_RCPP
  return Rcpp::wrap(nvforest_model_info(model));
  END_RCPP
}

RcppExport SEXP _cuda_ml_nvforest_cpu_predict(SEXP model, SEXP input_sexp,
                                              SEXP prediction_type_sexp,
                                              SEXP threshold_sexp,
                                              SEXP chunk_size_sexp) {
  BEGIN_RCPP
  Rcpp::traits::input_parameter<Rcpp::NumericMatrix const &>::type input(
      input_sexp);
  return Rcpp::wrap(nvforest_predict(
      model, input, Rcpp::as<int>(prediction_type_sexp),
      Rcpp::as<double>(threshold_sexp), Rcpp::as<int>(chunk_size_sexp)));
  END_RCPP
}

RcppExport SEXP _cuda_ml_nvforest_cpu_serialize(SEXP model) {
  BEGIN_RCPP
  return Rcpp::wrap(nvforest_serialize(model));
  END_RCPP
}

RcppExport SEXP _cuda_ml_nvforest_cpu_unserialize(
    SEXP bytes_sexp, SEXP device_sexp, SEXP device_id_sexp, SEXP layout_sexp,
    SEXP precision_sexp, SEXP chunk_size_sexp, SEXP align_bytes_sexp,
    SEXP averaged_vector_leaf_probabilities_sexp) {
  BEGIN_RCPP
  Rcpp::traits::input_parameter<Rcpp::RawVector const &>::type bytes(
      bytes_sexp);
  return Rcpp::wrap(nvforest_unserialize(
      bytes, Rcpp::as<int>(device_sexp), Rcpp::as<int>(device_id_sexp),
      Rcpp::as<int>(layout_sexp), Rcpp::as<int>(precision_sexp),
      Rcpp::as<int>(chunk_size_sexp), Rcpp::as<int>(align_bytes_sexp),
      Rcpp::as<bool>(averaged_vector_leaf_probabilities_sexp)));
  END_RCPP
}
