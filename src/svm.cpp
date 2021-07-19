#if HAS_CUML

#include "async_utils.h"
#include "cuda_utils.h"
#include "handle_utils.h"
#include "matrix_utils.h"
#include "pinned_host_vector.h"
#include "preprocessor.h"
#include "stream_allocator.h"

#include <thrust/async/copy.h>
#include <thrust/device_vector.h>
#include <cuml/svm/svc.hpp>

#include <memory>
#include <vector>

#else

#include "warn_cuml_missing.h"

#endif

#include <Rcpp.h>

// [[Rcpp::export(".svc_fit")]]
SEXP svc_fit(Rcpp::NumericMatrix const& x, Rcpp::NumericVector const& labels,
             Rcpp::NumericVector const& sample_weights, double const C,
             double const cache_size, int const max_iter,
             int const nochange_steps, double const tol, int const verbosity,
             double const epsilon, int const kernel_type, int const degree,
             double const gamma, double const coef0) {
#if HAS_CUML
  auto const m = cuml4r::Matrix<>(x, /*transpose=*/true);
  auto const n_samples = m.numCols;
  auto const n_features = m.numRows;

  auto stream_view = cuml4r::stream_allocator::getOrCreateStream();
  raft::handle_t handle;
  cuml4r::handle_utils::initializeHandle(handle, stream_view.value());

  // SVM input
  auto const& h_input = m.values;
  thrust::device_vector<double> d_input(h_input.size());
  auto CUML4R_ANONYMOUS_VARIABLE(input_h2d) = cuml4r::async_copy(
    stream_view.value(), h_input.cbegin(), h_input.cend(), d_input.begin());

  cuml4r::pinned_host_vector<double> h_labels(
    Rcpp::as<std::vector<double>>(labels));
  thrust::device_vector<double> d_labels(h_labels.size());
  auto CUML4R_ANONYMOUS_VARIABLE(labels_h2d) = cuml4r::async_copy(
    stream_view.value(), h_labels.cbegin(), h_labels.cend(), d_labels.begin());

  thrust::device_vector<double> d_sample_weights;
  cuml4r::unique_marker sample_weights_h2d;
  if (sample_weights.size() > 0) {
    cuml4r::pinned_host_vector<double> h_sample_weights(
      Rcpp::as<std::vector<double>>(sample_weights));
    d_sample_weights.resize(h_sample_weights.size());
    sample_weights_h2d =
      cuml4r::async_copy(stream_view.value(), h_sample_weights.cbegin(),
                         h_sample_weights.cend(), d_sample_weights.begin());
  }

  MLCommon::Matrix::KernelParams kernel_params{
    /*kernel=*/static_cast<MLCommon::Matrix::KernelType>(kernel_type), degree,
    gamma, coef0};

  // SVM output
  auto svc = std::make_unique<ML::SVM::SVC<double>>(
    handle, C, tol, kernel_params, cache_size, max_iter, nochange_steps,
    verbosity);

  svc->fit(d_input.data().get(), /*nrows=*/n_samples, /*ncols=*/n_features,
           d_labels.data().get(),
           d_sample_weights.empty() ? nullptr : d_sample_weights.data().get());

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  return Rcpp::XPtr<ML::SVM::SVC<double>>(svc.release());
#else

#include "warn_cuml_missing.h"

  return Rcpp::List();

#endif
}

// [[Rcpp::export(".svc_predict")]]
Rcpp::NumericVector svc_predict(SEXP svc_xptr,
                                Rcpp::NumericMatrix const& x) {
#if HAS_CUML
  auto const m = cuml4r::Matrix<>(x, /*transpose=*/true);
  auto const n_samples = m.numCols;
  auto const n_features = m.numRows;

  auto stream_view = cuml4r::stream_allocator::getOrCreateStream();
  raft::handle_t handle;
  cuml4r::handle_utils::initializeHandle(handle, stream_view.value());

  auto svc = Rcpp::XPtr<ML::SVM::SVC<double>>(svc_xptr);

  // input
  auto const& h_input = m.values;
  thrust::device_vector<double> d_input(h_input.size());
  auto CUML4R_ANONYMOUS_VARIABLE(input_h2d) = cuml4r::async_copy(
    stream_view.value(), h_input.cbegin(), h_input.cend(), d_input.begin());

  // output
  thrust::device_vector<double> d_preds(n_samples);

  svc->predict(d_input.data().get(), /*n_rows=*/n_samples, /*c_cols=*/n_features, d_preds.data().get());


  cuml4r::pinned_host_vector<double> h_preds(n_samples);
  auto CUML4R_ANONYMOUS_VARIABLE(preds_d2h) = cuml4r::async_copy(
    stream_view.value(), d_preds.cbegin(), d_preds.cend(), h_preds.begin());

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  return Rcpp::NumericVector(h_preds.begin(), h_preds.end());
#else

#include "warn_cuml_missing.h"

  return {};

#endif
}
