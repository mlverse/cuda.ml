#include "async_utils.cuh"
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

#include <Rcpp.h>

#include <memory>
#include <vector>

namespace {

namespace detail {

constexpr auto kSvcKernelParams = "kernel_params";
constexpr auto kKernelParamsType = "type";
constexpr auto kKernelParamsDegree = "degree";
constexpr auto kKernelParamsGamma = "gamma";
constexpr auto kKernelParamsCoef0 = "coef0";
constexpr auto kSvcSvmParams = "svm_params";
constexpr auto kSvmParamsC = "C";
constexpr auto kSvmParamsCacheSize = "cache_size";
constexpr auto kSvmParamsMaxIter = "max_iter";
constexpr auto kSvmParamsNoChangeSteps = "nochange_steps";
constexpr auto kSvmParamsTol = "tol";
constexpr auto kSvmParamsVerbosity = "verbosity";
constexpr auto kSvmParamsEpsilon = "epsilon";
constexpr auto kSvmParamsType = "type";
constexpr auto kSvcModel = "model";
constexpr auto kSvmModelNumSupportVectors = "n_support";
constexpr auto kSvmModelNumCols = "n_cols";
constexpr auto kSvmModelB = "b";
constexpr auto kSvmModelDualCoefs = "dual_coefs";
constexpr auto kSvmModelSupportVectors = "x_support";
constexpr auto kSvmModelSupportIdxes = "support_idx";
constexpr auto kSvmModelNumClasses = "n_classes";
constexpr auto kSvmModelUniqueLabels = "unique_labels";

template <typename T>
__host__ Rcpp::List getState(T const&) = delete;

template <typename T>
__host__ Rcpp::List getState(T const&, raft::handle_t const&) = delete;

template <>
__host__ Rcpp::List getState(
  MLCommon::Matrix::KernelParams const& kernel_params) {
  Rcpp::List state;

  state[kKernelParamsType] = static_cast<int>(kernel_params.kernel);
  state[kKernelParamsDegree] = kernel_params.degree;
  state[kKernelParamsGamma] = kernel_params.gamma;
  state[kKernelParamsCoef0] = kernel_params.coef0;

  return state;
}

template <>
__host__ Rcpp::List getState(ML::SVM::svmParameter const& svm_params) {
  Rcpp::List state;

  state[kSvmParamsC] = svm_params.C;
  state[kSvmParamsCacheSize] = svm_params.cache_size;
  state[kSvmParamsMaxIter] = svm_params.max_iter;
  state[kSvmParamsNoChangeSteps] = svm_params.nochange_steps;
  state[kSvmParamsTol] = svm_params.tol;
  state[kSvmParamsVerbosity] = svm_params.verbosity;
  state[kSvmParamsEpsilon] = svm_params.epsilon;
  state[kSvmParamsType] = static_cast<int>(svm_params.svmType);

  return state;
}

template <>
__host__ Rcpp::List getState(ML::SVM::svmModel<double> const& svm_model,
                             raft::handle_t const& handle) {
  Rcpp::List state;
  auto* const stream = handle.get_stream();

  cuml4r::pinned_host_vector<double> h_dual_coefs(svm_model.n_support);
  CUDA_RT_CALL(cudaMemcpyAsync(
    /*dst=*/h_dual_coefs.data(),
    /*src=*/svm_model.dual_coefs,
    /*count=*/svm_model.n_support * sizeof(double),
    /*kind=*/cudaMemcpyDeviceToHost, stream));

  cuml4r::pinned_host_vector<double> h_x_support(svm_model.n_support *
                                                 svm_model.n_cols);
  CUDA_RT_CALL(cudaMemcpyAsync(
    /*dst=*/h_x_support.data(),
    /*src=*/svm_model.x_support,
    /*count=*/svm_model.n_support * svm_model.n_cols * sizeof(double),
    /*kind=*/cudaMemcpyDeviceToHost, stream));

  cuml4r::pinned_host_vector<int> h_support_idx(svm_model.n_support);
  CUDA_RT_CALL(cudaMemcpyAsync(
    /*dst=*/h_support_idx.data(),
    /*src=*/svm_model.support_idx,
    /*count=*/svm_model.n_support * sizeof(int),
    /*kind=*/cudaMemcpyDeviceToHost, stream));

  cuml4r::pinned_host_vector<double> h_unique_labels(svm_model.n_classes);
  CUDA_RT_CALL(cudaMemcpyAsync(
    /*dst=*/h_unique_labels.data(),
    /*src=*/svm_model.unique_labels,
    /*count=*/svm_model.n_classes * sizeof(double),
    /*kind=*/cudaMemcpyDeviceToHost, stream));

  CUDA_RT_CALL(cudaStreamSynchronize(stream));

  state[kSvmModelNumSupportVectors] = svm_model.n_support;
  state[kSvmModelNumCols] = svm_model.n_cols;
  state[kSvmModelB] = svm_model.b;
  state[kSvmModelDualCoefs] =
    Rcpp::NumericVector(h_dual_coefs.begin(), h_dual_coefs.end());
  state[kSvmModelSupportVectors] =
    Rcpp::NumericVector(h_x_support.begin(), h_x_support.end());
  state[kSvmModelSupportIdxes] =
    Rcpp::IntegerVector(h_support_idx.begin(), h_support_idx.end());
  state[kSvmModelNumClasses] = svm_model.n_classes;
  state[kSvmModelUniqueLabels] =
    Rcpp::NumericVector(h_unique_labels.begin(), h_unique_labels.end());

  return state;
}

template <typename T>
__host__ void setState(T&, Rcpp::List const&) = delete;

template <typename T>
__host__ void setState(T&, raft::handle_t const&, Rcpp::List const&) = delete;

template <>
__host__ void setState(MLCommon::Matrix::KernelParams& kernel_params,
                       Rcpp::List const& state) {
  kernel_params.kernel = static_cast<MLCommon::Matrix::KernelType>(
    Rcpp::as<int>(state[kKernelParamsType]));
  kernel_params.degree = state[kKernelParamsDegree];
  kernel_params.gamma = state[kKernelParamsGamma];
  kernel_params.coef0 = state[kKernelParamsCoef0];
}

template <>
__host__ void setState(ML::SVM::svmParameter& svm_params,
                       Rcpp::List const& state) {
  svm_params.C = state[kSvmParamsC];
  svm_params.cache_size = state[kSvmParamsCacheSize];
  svm_params.max_iter = state[kSvmParamsMaxIter];
  svm_params.nochange_steps = state[kSvmParamsNoChangeSteps];
  svm_params.tol = state[kSvmParamsTol];
  svm_params.verbosity = state[kSvmParamsVerbosity];
  svm_params.epsilon = state[kSvmParamsEpsilon];
  svm_params.svmType =
    static_cast<ML::SVM::SvmType>(Rcpp::as<int>(state[kSvmParamsType]));
}

template <>
__host__ void setState(ML::SVM::svmModel<double>& svm_model,
                       raft::handle_t const& handle, Rcpp::List const& state) {
  int const n_support = state[kSvmModelNumSupportVectors];
  int const n_cols = state[kSvmModelNumCols];

  svm_model.n_support = n_support;
  svm_model.n_cols = n_cols;
  svm_model.b = state[kSvmModelB];

  auto const stream_view = handle.get_stream_view();

  CUDA_RT_CALL(cudaMallocAsync(
    &svm_model.dual_coefs, n_support * sizeof(double), stream_view.value()));
  auto const h_dual_coefs =
    Rcpp::as<cuml4r::pinned_host_vector<double>>(state[kSvmModelDualCoefs]);
  CUDA_RT_CALL(cudaMemcpyAsync(
    /*dst=*/svm_model.dual_coefs,
    /*src=*/h_dual_coefs.data(),
    /*count=*/n_support * sizeof(double),
    /*kind=*/cudaMemcpyHostToDevice,
    /*stream=*/stream_view.value()));

  CUDA_RT_CALL(cudaMallocAsync(&svm_model.x_support,
                               n_support * n_cols * sizeof(double),
                               stream_view.value()));
  auto const h_x_support = Rcpp::as<cuml4r::pinned_host_vector<double>>(
    state[kSvmModelSupportVectors]);
  CUDA_RT_CALL(cudaMemcpyAsync(
    /*dst=*/svm_model.x_support,
    /*src=*/h_x_support.data(),
    /*count=*/n_support * n_cols * sizeof(double),
    /*kind=*/cudaMemcpyHostToDevice,
    /*stream=*/stream_view.value()));

  CUDA_RT_CALL(cudaMallocAsync(&svm_model.support_idx, n_support * sizeof(int),
                               stream_view.value()));
  auto const h_support_idx =
    Rcpp::as<cuml4r::pinned_host_vector<int>>(state[kSvmModelSupportIdxes]);
  CUDA_RT_CALL(cudaMemcpyAsync(
    /*dst=*/svm_model.support_idx,
    /*src=*/h_support_idx.data(),
    /*count=*/n_support * sizeof(int),
    /*kind=*/cudaMemcpyHostToDevice,
    /*stream=*/stream_view.value()));

  int const n_classes = state[kSvmModelNumClasses];
  svm_model.n_classes = n_classes;

  CUDA_RT_CALL(cudaMallocAsync(
    &svm_model.unique_labels, n_classes * sizeof(double), stream_view.value()));
  auto const h_unique_labels =
    Rcpp::as<cuml4r::pinned_host_vector<double>>(state[kSvmModelUniqueLabels]);
  CUDA_RT_CALL(cudaMemcpyAsync(
    /*dst=*/svm_model.unique_labels,
    /*src=*/h_unique_labels.data(),
    /*count=*/n_classes * sizeof(double),
    /*kind=*/cudaMemcpyHostToDevice,
    /*stream=*/stream_view.value()));

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));
}

}  // namespace detail

class ModelCtx {
 public:
  using model_t = ML::SVM::SVC<double>;

  // model object must be destroyed first
  std::unique_ptr<raft::handle_t> const handle_;
  std::unique_ptr<model_t> const model_;

  __host__ ModelCtx(std::unique_ptr<raft::handle_t> handle,
                    std::unique_ptr<model_t> model) noexcept
    : handle_(std::move(handle)), model_(std::move(model)) {}

  __host__ Rcpp::List getState() const {
    Rcpp::List state;

    state[detail::kSvcKernelParams] = detail::getState(model_->kernel_params);
    state[detail::kSvcSvmParams] = detail::getState(model_->param);
    state[detail::kSvcModel] = detail::getState(model_->model, *handle_);

    return state;
  }

  __host__ void setState(Rcpp::List const& state) {
    detail::setState(model_->kernel_params, state[detail::kSvcKernelParams]);
    detail::setState(model_->param, state[detail::kSvcSvmParams]);
    detail::setState(model_->model, *handle_, state[detail::kSvcModel]);
  }
};

}  // namespace

namespace cuml4r {

__host__ SEXP svc_fit(Rcpp::NumericMatrix const& input,
                      Rcpp::NumericVector const& labels, double const cost,
                      int const kernel, double const gamma, double const coef0,
                      int const degree, double const tol, int const max_iter,
                      int const nochange_steps, double const cache_size,
                      Rcpp::NumericVector const& sample_weights,
                      int const verbosity) {
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

  auto h_labels(Rcpp::as<cuml4r::pinned_host_vector<double>>(labels));
  thrust::device_vector<double> d_labels(h_labels.size());
  auto CUML4R_ANONYMOUS_VARIABLE(labels_h2d) = cuml4r::async_copy(
    stream_view.value(), h_labels.cbegin(), h_labels.cend(), d_labels.begin());

  thrust::device_vector<double> d_sample_weights;
  cuml4r::unique_marker sample_weights_h2d;
  if (sample_weights.size() > 0) {
    auto const h_sample_weights(
      Rcpp::as<cuml4r::pinned_host_vector<double>>(sample_weights));
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

  return Rcpp::XPtr<ModelCtx>(new ModelCtx(std::move(handle), std::move(svc)));
}

__host__ SEXP svc_predict(SEXP model_xptr, Rcpp::NumericMatrix const& input,
                          bool predict_class) {
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
}

__host__ Rcpp::List svc_get_state(SEXP model) {
  return Rcpp::XPtr<ModelCtx>(model)->getState();
}

__host__ SEXP svc_set_state(Rcpp::List const& state) {
  auto stream_view = cuml4r::stream_allocator::getOrCreateStream();
  auto handle = std::make_unique<raft::handle_t>();
  cuml4r::handle_utils::initializeHandle(*handle, stream_view.value());

  auto model = std::make_unique<ML::SVM::SVC<double>>(*handle);

  auto model_ctx = std::make_unique<ModelCtx>(
    /*handle=*/std::move(handle), /*model=*/std::move(model));
  model_ctx->setState(state);

  return Rcpp::XPtr<ModelCtx>(model_ctx.release());
}

}  // namespace cuml4r
