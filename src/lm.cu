#include "async_utils.cuh"
#include "cuda_utils.h"
#include "handle_utils.h"
#include "lm_constants.h"
#include "lm_params.h"
#include "matrix_utils.h"
#include "pinned_host_vector.h"
#include "preprocessor.h"
#include "stream_allocator.h"

#include <thrust/device_vector.h>
#include <cuml/version_config.hpp>

#include <Rcpp.h>

#include <cmath>
#include <vector>

namespace cuml4r {

__host__ Rcpp::List lm_fit(
  Rcpp::NumericMatrix const& x, Rcpp::NumericVector const& y,
  lm::InterceptType const intercept_type, bool const fit_intercept,
  bool const normalize_input,
  std::function<void(raft::handle_t&, lm::Params const&)> const& fit_impl) {
  auto const m = Matrix<>(x, /*transpose=*/true);
  auto const n_rows = m.numCols;
  auto const n_cols = m.numRows;

#if (CUML4R_LIBCUML_VERSION(CUML_VERSION_MAJOR, CUML_VERSION_MINOR) >= \
     CUML4R_LIBCUML_VERSION(24, 0))
  auto h_normalized_input = pinned_host_vector<double>();
  auto normalization_means = std::vector<double>();
  auto normalization_norms = std::vector<double>();
  auto const use_host_normalization = normalize_input && fit_intercept;
  if (use_host_normalization) {
    h_normalized_input = m.values;
    normalization_means.resize(n_cols);
    normalization_norms.resize(n_cols);

    for (size_t col = 0; col < n_cols; ++col) {
      auto const offset = col * n_rows;
      double mean = 0;
      for (size_t row = 0; row < n_rows; ++row) {
        mean += h_normalized_input[offset + row];
      }
      mean /= n_rows;
      normalization_means[col] = mean;

      double squared_norm = 0;
      for (size_t row = 0; row < n_rows; ++row) {
        auto& value = h_normalized_input[offset + row];
        value -= mean;
        squared_norm += value * value;
      }

      auto const norm = std::sqrt(squared_norm);
      normalization_norms[col] = norm;
      if (norm != 0) {
        for (size_t row = 0; row < n_rows; ++row) {
          h_normalized_input[offset + row] /= norm;
        }
      }
    }
  }
#endif

  auto stream_view = stream_allocator::getOrCreateStream();
  raft::handle_t handle;
  handle_utils::initializeHandle(handle, stream_view.value());

  // LM input
  auto const* h_input = &m.values;
#if (CUML4R_LIBCUML_VERSION(CUML_VERSION_MAJOR, CUML_VERSION_MINOR) >= \
     CUML4R_LIBCUML_VERSION(24, 0))
  if (use_host_normalization) {
    h_input = &h_normalized_input;
  }
#endif
  thrust::device_vector<double> d_input(h_input->size());
  auto CUML4R_ANONYMOUS_VARIABLE(input_h2d) = async_copy(
    stream_view.value(), h_input->cbegin(), h_input->cend(), d_input.begin());
  auto const h_labels = Rcpp::as<pinned_host_vector<double>>(y);
  thrust::device_vector<double> d_labels(h_labels.size());
  auto CUML4R_ANONYMOUS_VARIABLE(labels_h2d) = async_copy(
    stream_view.value(), h_labels.cbegin(), h_labels.cend(), d_labels.begin());

  // LM output
  thrust::device_vector<double> d_coef(n_cols);
  thrust::device_vector<double> d_intercept;
  double h_intercept = 0;

  lm::Params params;
  params.d_input = d_input.data().get();
  params.n_rows = n_rows;
  params.n_cols = n_cols;
  params.d_labels = d_labels.data().get();
  params.d_coef = d_coef.data().get();
  switch (intercept_type) {
    case lm::InterceptType::HOST: {
      params.intercept = &h_intercept;
      break;
    }
    case lm::InterceptType::DEVICE: {
      d_intercept.resize(1);
      params.intercept = d_intercept.data().get();
      break;
    }
  }
  params.fit_intercept = fit_intercept;
  params.normalize_input = normalize_input;

  fit_impl(handle, params);

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  pinned_host_vector<double> h_coef(n_cols);
  auto CUML4R_ANONYMOUS_VARIABLE(coef_d2h) = async_copy(
    stream_view.value(), d_coef.cbegin(), d_coef.cend(), h_coef.begin());
  switch (intercept_type) {
    case lm::InterceptType::HOST: {
      break;
    }
    case lm::InterceptType::DEVICE: {
      h_intercept = d_intercept[0];
      break;
    }
  }

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

#if (CUML4R_LIBCUML_VERSION(CUML_VERSION_MAJOR, CUML_VERSION_MINOR) >= \
     CUML4R_LIBCUML_VERSION(24, 0))
  if (use_host_normalization) {
    for (size_t col = 0; col < n_cols; ++col) {
      if (normalization_norms[col] != 0) {
        h_coef[col] /= normalization_norms[col];
      }
      h_intercept -= normalization_means[col] * h_coef[col];
    }
  }
#endif

  Rcpp::List model;
  model[cuml4r::lm::kCoef] = Rcpp::NumericVector(h_coef.begin(), h_coef.end());
  model[cuml4r::lm::kIntercept] = h_intercept;

  return model;
}

}  // namespace cuml4r
