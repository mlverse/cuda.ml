#include "async_utils.cuh"
#include "cuda_utils.h"
#include "handle_utils.h"
#include "pinned_host_vector.h"
#include "stream_allocator.h"
#include "svm_serde.h"

namespace cuml4r {
namespace detail {

__host__ Rcpp::List getState(
  MLCommon::Matrix::KernelParams const& kernel_params) {
  Rcpp::List state;

  state[kKernelParamsType] = static_cast<int>(kernel_params.kernel);
  state[kKernelParamsDegree] = kernel_params.degree;
  state[kKernelParamsGamma] = kernel_params.gamma;
  state[kKernelParamsCoef0] = kernel_params.coef0;

  return state;
}

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

__host__ Rcpp::List getState(ML::SVM::svmModel<double> const& svm_model,
                             raft::handle_t const& handle) {
  Rcpp::List state;
  auto* const stream = handle.get_stream();

  pinned_host_vector<double> h_dual_coefs(svm_model.n_support);
  CUDA_RT_CALL(cudaMemcpyAsync(
    /*dst=*/h_dual_coefs.data(),
    /*src=*/svm_model.dual_coefs,
    /*count=*/svm_model.n_support * sizeof(double),
    /*kind=*/cudaMemcpyDeviceToHost, stream));

  pinned_host_vector<double> h_x_support(svm_model.n_support *
                                         svm_model.n_cols);
  CUDA_RT_CALL(cudaMemcpyAsync(
    /*dst=*/h_x_support.data(),
    /*src=*/svm_model.x_support,
    /*count=*/svm_model.n_support * svm_model.n_cols * sizeof(double),
    /*kind=*/cudaMemcpyDeviceToHost, stream));

  pinned_host_vector<int> h_support_idx(svm_model.n_support);
  CUDA_RT_CALL(cudaMemcpyAsync(
    /*dst=*/h_support_idx.data(),
    /*src=*/svm_model.support_idx,
    /*count=*/svm_model.n_support * sizeof(int),
    /*kind=*/cudaMemcpyDeviceToHost, stream));

  pinned_host_vector<double> h_unique_labels(svm_model.n_classes);
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

__host__ void setState(MLCommon::Matrix::KernelParams& kernel_params,
                       Rcpp::List const& state) {
  kernel_params.kernel = static_cast<MLCommon::Matrix::KernelType>(
    Rcpp::as<int>(state[kKernelParamsType]));
  kernel_params.degree = state[kKernelParamsDegree];
  kernel_params.gamma = state[kKernelParamsGamma];
  kernel_params.coef0 = state[kKernelParamsCoef0];
}

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

__host__ void setState(ML::SVM::svmModel<double>& svm_model,
                       raft::handle_t const& handle, Rcpp::List const& state) {
  int const n_support = state[kSvmModelNumSupportVectors];
  int const n_cols = state[kSvmModelNumCols];

  svm_model.n_support = n_support;
  svm_model.n_cols = n_cols;
  svm_model.b = state[kSvmModelB];

  auto const stream_view = handle.get_stream_view();

  CUDA_RT_CALL(cudaMalloc(&svm_model.dual_coefs, n_support * sizeof(double)));
  auto const h_dual_coefs =
    Rcpp::as<pinned_host_vector<double>>(state[kSvmModelDualCoefs]);
  CUDA_RT_CALL(cudaMemcpyAsync(
    /*dst=*/svm_model.dual_coefs,
    /*src=*/h_dual_coefs.data(),
    /*count=*/n_support * sizeof(double),
    /*kind=*/cudaMemcpyHostToDevice,
    /*stream=*/stream_view.value()));

  CUDA_RT_CALL(
    cudaMalloc(&svm_model.x_support, n_support * n_cols * sizeof(double)));
  auto const h_x_support =
    Rcpp::as<pinned_host_vector<double>>(state[kSvmModelSupportVectors]);
  CUDA_RT_CALL(cudaMemcpyAsync(
    /*dst=*/svm_model.x_support,
    /*src=*/h_x_support.data(),
    /*count=*/n_support * n_cols * sizeof(double),
    /*kind=*/cudaMemcpyHostToDevice,
    /*stream=*/stream_view.value()));

  CUDA_RT_CALL(cudaMalloc(&svm_model.support_idx, n_support * sizeof(int)));
  auto const h_support_idx =
    Rcpp::as<pinned_host_vector<int>>(state[kSvmModelSupportIdxes]);
  CUDA_RT_CALL(cudaMemcpyAsync(
    /*dst=*/svm_model.support_idx,
    /*src=*/h_support_idx.data(),
    /*count=*/n_support * sizeof(int),
    /*kind=*/cudaMemcpyHostToDevice,
    /*stream=*/stream_view.value()));

  int const n_classes = state[kSvmModelNumClasses];
  svm_model.n_classes = n_classes;

  CUDA_RT_CALL(
    cudaMalloc(&svm_model.unique_labels, n_classes * sizeof(double)));
  auto const h_unique_labels =
    Rcpp::as<pinned_host_vector<double>>(state[kSvmModelUniqueLabels]);
  CUDA_RT_CALL(cudaMemcpyAsync(
    /*dst=*/svm_model.unique_labels,
    /*src=*/h_unique_labels.data(),
    /*count=*/n_classes * sizeof(double),
    /*kind=*/cudaMemcpyHostToDevice,
    /*stream=*/stream_view.value()));

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));
}

}  // namespace detail
}  // namespace cuml4r
