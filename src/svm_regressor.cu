#include "async_utils.cuh"
#include "cuda_utils.h"
#include "handle_utils.h"
#include "matrix_utils.h"
#include "pinned_host_vector.h"
#include "preprocessor.h"
#include "stream_allocator.h"
#include "svm_serde.h"

#include <cuml/svm/svm_parameter.h>
#include <thrust/async/copy.h>
#include <thrust/device_vector.h>
#include <cuml/svm/svc.hpp>
#include <cuml/svm/svr.hpp>

#include <Rcpp.h>

#include <memory>

namespace cuml4r {

namespace {

constexpr auto kSvrModel = "model";
constexpr auto kSvrKernelParams = "kernel_params";
constexpr auto kSvrCacheSize = "cache_size";

struct SVR {
  std::unique_ptr<raft::handle_t> const handle_;
  std::unique_ptr<ML::SVM::svmModel<double>> const model_;
  MLCommon::Matrix::KernelParams kernelParams_;
  double cacheSize_;

  __host__ SVR(std::unique_ptr<raft::handle_t> handle,
               std::unique_ptr<ML::SVM::svmModel<double>> model,
               MLCommon::Matrix::KernelParams kernel_params,
               double const cache_size) noexcept
    : handle_(std::move(handle)),
      model_(std::move(model)),
      kernelParams_(std::move(kernel_params)),
      cacheSize_(cache_size) {}

  __host__ Rcpp::List getState() const {
    Rcpp::List state;

    state[kSvrModel] =
      detail::getState(/*svm_model=*/*model_, /*handle=*/*handle_);
    state[kSvrKernelParams] = detail::getState(kernelParams_);
    state[kSvrCacheSize] = cacheSize_;

    return state;
  }

  __host__ void setState(Rcpp::List const& state) {
    detail::setState(/*svm_model=*/*model_, /*handle=*/*handle_,
                     /*state=*/state[kSvrModel]);
    detail::setState(/*kernel_params=*/kernelParams_,
                     /*state=*/state[kSvrKernelParams]);
    cacheSize_ = state[kSvrCacheSize];
  }
};

}  // namespace

__host__ SEXP svr_fit(Rcpp::NumericMatrix const& X,
                      Rcpp::NumericVector const& y, double const cost,
                      int const kernel, double const gamma, double const coef0,
                      int const degree, double const tol, int const max_iter,
                      int const nochange_steps, double const cache_size,
                      double epsilon, Rcpp::NumericVector const& sample_weights,
                      int const verbosity) {
  auto const m = Matrix<>(X, /*transpose=*/true);
  auto const n_samples = m.numCols;
  auto const n_features = m.numRows;

  auto stream_view = stream_allocator::getOrCreateStream();
  auto handle = std::make_unique<raft::handle_t>();
  handle_utils::initializeHandle(*handle, stream_view.value());

  // SVM input
  auto const& h_X = m.values;
  thrust::device_vector<double> d_X(h_X.size());
  auto CUML4R_ANONYMOUS_VARIABLE(X_h2d) =
    async_copy(stream_view.value(), h_X.cbegin(), h_X.cend(), d_X.begin());

  auto const h_y(Rcpp::as<pinned_host_vector<double>>(y));
  thrust::device_vector<double> d_y(h_y.size());
  auto CUML4R_ANONYMOUS_VARIABLE(y_h2d) =
    async_copy(stream_view.value(), h_y.cbegin(), h_y.cend(), d_y.begin());

  thrust::device_vector<double> d_sample_weights;
  AsyncCopyCtx sample_weights_h2d;
  if (sample_weights.size() > 0) {
    auto const h_sample_weights(
      Rcpp::as<pinned_host_vector<double>>(sample_weights));
    d_sample_weights.resize(h_sample_weights.size());
    sample_weights_h2d =
      async_copy(stream_view.value(), h_sample_weights.cbegin(),
                 h_sample_weights.cend(), d_sample_weights.begin());
  }

  ML::SVM::svmParameter param;
  param.C = cost;
  param.cache_size = cache_size, param.max_iter = max_iter;
  param.nochange_steps = nochange_steps;
  param.tol = tol;
  param.verbosity = verbosity;
  param.epsilon = epsilon;
  param.svmType = ML::SVM::SvmType::EPSILON_SVR;
  MLCommon::Matrix::KernelParams kernel_params{
    /*kernel=*/static_cast<MLCommon::Matrix::KernelType>(kernel), degree, gamma,
    coef0};

  // SVM output
  auto model = std::make_unique<ML::SVM::svmModel<double>>();

  ML::SVM::svrFit(
    *handle, d_X.data().get(),
    /*n_rows=*/n_samples,
    /*n_cols=*/n_features, d_y.data().get(), param, kernel_params, *model,
    d_sample_weights.empty() ? nullptr : d_sample_weights.data().get());

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  auto svr = std::make_unique<SVR>(
    /*handle=*/std::move(handle),
    /*model=*/std::move(model),
    /*kernel_params=*/std::move(kernel_params),
    /*cache_size=*/cache_size);

  return Rcpp::XPtr<SVR>(svr.release());
}

__host__ Rcpp::NumericVector svr_predict(SEXP svr_xptr,
                                         Rcpp::NumericMatrix const& X) {
  auto const m = Matrix<>(X, /*transpose=*/true);
  auto const n_samples = m.numCols;
  auto const n_features = m.numRows;

  auto const svr = Rcpp::XPtr<SVR>(svr_xptr).get();

  auto stream_view = stream_allocator::getOrCreateStream();
  raft::handle_t handle;
  handle_utils::initializeHandle(handle, stream_view.value());

  // input
  auto const& h_X = m.values;
  thrust::device_vector<double> d_X(h_X.size());
  auto CUML4R_ANONYMOUS_VARIABLE(X_h2d) =
    async_copy(stream_view.value(), h_X.cbegin(), h_X.cend(), d_X.begin());

  // output
  thrust::device_vector<double> d_y(n_samples);

  ML::SVM::svcPredict(handle, /*input=*/d_X.data().get(), /*n_rows=*/n_samples,
                      /*c_cols=*/n_features,
                      /*kernel_params=*/svr->kernelParams_,
                      /*model=*/*svr->model_, /*preds=*/d_y.data().get(),
                      /*cache_size=*/svr->cacheSize_, /*predict_class=*/false);

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  pinned_host_vector<double> h_y(n_samples);
  auto CUML4R_ANONYMOUS_VARIABLE(y_d2h) =
    async_copy(stream_view.value(), d_y.cbegin(), d_y.cend(), h_y.begin());
  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  return Rcpp::NumericVector(h_y.begin(), h_y.end());
}

__host__ Rcpp::List svr_get_state(SEXP model) {
  return Rcpp::XPtr<SVR>(model)->getState();
}

__host__ SEXP svr_set_state(Rcpp::List const& state) {
  auto stream_view = stream_allocator::getOrCreateStream();
  auto handle = std::make_unique<raft::handle_t>();
  handle_utils::initializeHandle(*handle, stream_view.value());

  auto model = std::make_unique<SVR>(
    /*handle=*/std::move(handle),
    /*model=*/std::make_unique<ML::SVM::svmModel<double>>(),
    /*kernel_params=*/MLCommon::Matrix::KernelParams(),
    /*cache_size=*/0);
  model->setState(state);

  return Rcpp::XPtr<SVR>(model.release());
}

}  // namespace cuml4r
