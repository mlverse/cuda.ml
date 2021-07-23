#if HAS_CUML

#include "async_utils.h"
#include "cuda_utils.h"
#include "handle_utils.h"
#include "matrix_utils.h"
#include "pinned_host_vector.h"
#include "preprocessor.h"
#include "stream_allocator.h"

#include <cuml/svm/svm_parameter.h>
#include <thrust/async/copy.h>
#include <thrust/device_vector.h>
#include <cuml/svm/svc.hpp>

#include <memory>
#include <vector>

#else

#include "warn_cuml_missing.h"

#endif

#include <Rcpp.h>

#if HAS_CUML

namespace {

struct ModelCtx {
  using model_t = ML::SVM::SVC<double>;

  std::unique_ptr<model_t> const model_;
  std::unique_ptr<raft::handle_t> const handle_;

  ModelCtx(std::unique_ptr<model_t> model,
           std::unique_ptr<raft::handle_t> handle) noexcept
    : model_(std::move(model)), handle_(std::move(handle)) {}
};

}  // namespace

#else

#include "warn_cuml_missing.h"

#endif

// [[Rcpp::export(".svc_fit")]]
SEXP svc_fit(Rcpp::NumericMatrix const& input,
             Rcpp::NumericVector const& labels, double const cost,
             int const kernel, double const gamma, double const coef0,
             int const degree, double const tol, int const max_iter,
             int const nochange_steps, double const cache_size,
             Rcpp::NumericVector const& sample_weights, int const verbosity) {
#if HAS_CUML

  auto const m = cuml4r::Matrix<>(input, /*transpose=*/true);
  auto const n_samples = m.numCols;
  auto const n_features = m.numRows;

  auto stream_view = cuml4r::stream_allocator::getOrCreateStream();
  auto handle = std::make_unique<raft::handle_t>();
  cuml4r::handle_utils::initializeHandle(*handle, stream_view.value());

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
    /*kernel=*/static_cast<MLCommon::Matrix::KernelType>(kernel), degree, gamma,
    coef0};

  // SVM output
  auto svc = std::make_unique<ML::SVM::SVC<double>>(
    *handle, /*C=*/cost, tol, kernel_params, cache_size, max_iter,
    nochange_steps, verbosity);

  svc->fit(d_input.data().get(), /*nrows=*/n_samples, /*ncols=*/n_features,
           d_labels.data().get(),
           d_sample_weights.empty() ? nullptr : d_sample_weights.data().get());

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  return Rcpp::XPtr<ModelCtx>(new ModelCtx(std::move(svc), std::move(handle)));
#else

#include "warn_cuml_missing.h"

  return Rcpp::List();

#endif
}

// [[Rcpp::export(".svc_predict")]]
SEXP svc_predict(SEXP model_xptr,
                                Rcpp::NumericMatrix const& input,
                                bool predict_class) {
#if HAS_CUML
  auto const m = cuml4r::Matrix<>(input, /*transpose=*/true);
  int const n_samples = m.numCols;
  int const n_features = m.numRows;

  auto ctx = Rcpp::XPtr<ModelCtx>(model_xptr);
  auto const& svc = ctx->model_;
  auto* stream = ctx->handle_->get_stream();

  // input
  auto const& h_input = m.values;
  thrust::device_vector<double> d_input(h_input.size());
  auto CUML4R_ANONYMOUS_VARIABLE(input_h2d) = cuml4r::async_copy(
    stream, h_input.cbegin(), h_input.cend(), d_input.begin());

  // output
  thrust::device_vector<double> d_preds(n_samples);

  if (predict_class) {
    svc->predict(/*input=*/d_input.data().get(), /*n_rows=*/n_samples,
                 /*c_cols=*/n_features, /*preds=*/d_preds.data().get());
  } else {
    ML::SVM::svcPredict(
      /*handle=*/*ctx->handle_, /*input=*/d_input.data().get(),
      /*n_rows=*/n_samples,
      /*c_cols=*/n_features, /*kernel_parames=*/svc->kernel_params,
      /*model=*/svc->model, /*preds=*/d_preds.data().get(),
      /*buffer_size=*/svc->param.cache_size, /*predict_class=*/false);
  }

  cuml4r::pinned_host_vector<double> h_preds(n_samples);
  auto CUML4R_ANONYMOUS_VARIABLE(preds_d2h) = cuml4r::async_copy(
    stream, d_preds.cbegin(), d_preds.cend(), h_preds.begin());
  CUDA_RT_CALL(cudaStreamSynchronize(stream));

  return Rcpp::NumericVector(h_preds.begin(), h_preds.end());
#else

#include "warn_cuml_missing.h"

  return Rcpp::IntegerVector();

#endif
}
