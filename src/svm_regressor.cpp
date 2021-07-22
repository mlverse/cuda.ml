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
#include <cuml/svm/svr.hpp>

#include <memory>
#include <vector>

#else

#include "warn_cuml_missing.h"

#endif

#include <Rcpp.h>

#if HAS_CUML

namespace {

struct SVR {
  std::unique_ptr<ML::SVM::svmModel<double>> const model_;
  MLCommon::Matrix::KernelParams kernelParams_;
  double const cacheSize_;

  SVR(std::unique_ptr<ML::SVM::svmModel<double>> model,
      MLCommon::Matrix::KernelParams kernel_params, double const cache_size)
  noexcept
    : model_(std::move(model)),
      kernelParams_(std::move(kernel_params)),
      cacheSize_(cache_size) {}
};

}  // namespace

#else

#include "warn_cuml_missing.h"

#endif

// [[Rcpp::export(".svr_fit")]]
SEXP svr_fit(Rcpp::NumericMatrix const& X, Rcpp::NumericVector const& y,
             double const cost, int const kernel, double const gamma,
             double const coef0, int const degree, double const tol,
             int const max_iter, int const nochange_steps,
             double const cache_size, double epsilon,
             Rcpp::NumericVector const& sample_weights, int const verbosity) {
#if HAS_CUML
  auto const m = cuml4r::Matrix<>(X, /*transpose=*/true);
  auto const n_samples = m.numCols;
  auto const n_features = m.numRows;

  auto stream_view = cuml4r::stream_allocator::getOrCreateStream();
  raft::handle_t handle;
  cuml4r::handle_utils::initializeHandle(handle, stream_view.value());

  // SVM input
  auto const& h_X = m.values;
  thrust::device_vector<double> d_X(h_X.size());
  auto CUML4R_ANONYMOUS_VARIABLE(X_h2d) = cuml4r::async_copy(
    stream_view.value(), h_X.cbegin(), h_X.cend(), d_X.begin());

  cuml4r::pinned_host_vector<double> h_y(Rcpp::as<std::vector<double>>(y));
  thrust::device_vector<double> d_y(h_y.size());
  auto CUML4R_ANONYMOUS_VARIABLE(y_h2d) = cuml4r::async_copy(
    stream_view.value(), h_y.cbegin(), h_y.cend(), d_y.begin());

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

  ML::SVM::svmParameter param{
    /*C=*/cost, cache_size,
    max_iter,   nochange_steps,
    tol,        verbosity,
    epsilon,    /*svmType=*/ML::SVM::SvmType::EPSILON_SVR};
  MLCommon::Matrix::KernelParams kernel_params{
    /*kernel=*/static_cast<MLCommon::Matrix::KernelType>(kernel), degree, gamma,
    coef0};

  // SVM output
  auto model = std::make_unique<ML::SVM::svmModel<double>>();

  ML::SVM::svrFit(
    handle, d_X.data().get(),
    /*n_rows=*/n_samples,
    /*n_cols=*/n_features, d_y.data().get(), param, kernel_params, *model,
    d_sample_weights.empty() ? nullptr : d_sample_weights.data().get());

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  auto svr = std::make_unique<SVR>(std::move(model), std::move(kernel_params),
                                   cache_size);

  return Rcpp::XPtr<SVR>(svr.release());
#else

#include "warn_cuml_missing.h"

  return Rcpp::List();

#endif
}

// [[Rcpp::export(".svr_predict")]]
Rcpp::NumericVector svr_predict(SEXP svr_xptr, Rcpp::NumericMatrix const& X) {
#if HAS_CUML
  auto const m = cuml4r::Matrix<>(X, /*transpose=*/true);
  auto const n_samples = m.numCols;
  auto const n_features = m.numRows;

  auto const svr = Rcpp::XPtr<SVR>(svr_xptr).get();

  auto stream_view = cuml4r::stream_allocator::getOrCreateStream();
  raft::handle_t handle;
  cuml4r::handle_utils::initializeHandle(handle, stream_view.value());

  // input
  auto const& h_X = m.values;
  thrust::device_vector<double> d_X(h_X.size());
  auto CUML4R_ANONYMOUS_VARIABLE(X_h2d) = cuml4r::async_copy(
    stream_view.value(), h_X.cbegin(), h_X.cend(), d_X.begin());

  // output
  thrust::device_vector<double> d_y(n_samples);

  ML::SVM::svcPredict(handle, /*input=*/d_X.data().get(), /*n_rows=*/n_samples,
                      /*c_cols=*/n_features,
                      /*kernel_params=*/svr->kernelParams_,
                      /*model=*/*svr->model_, /*preds=*/d_y.data().get(),
                      /*cache_size=*/svr->cacheSize_, /*predict_class=*/false);

  cuml4r::pinned_host_vector<double> h_y(n_samples);
  auto CUML4R_ANONYMOUS_VARIABLE(y_d2h) = cuml4r::async_copy(
    stream_view.value(), d_y.cbegin(), d_y.cend(), h_y.begin());
  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  return Rcpp::NumericVector(h_y.begin(), h_y.end());
#else

#include "warn_cuml_missing.h"

  return {};

#endif
}
