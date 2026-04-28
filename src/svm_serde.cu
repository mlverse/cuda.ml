#include "async_utils.cuh"
#include "cuda_utils.h"
#include "handle_utils.h"
#include "pinned_host_vector.h"
#include "stream_allocator.h"
#include "svm_serde.h"

namespace cuml4r {
namespace detail {

namespace {

#if (CUML4R_LIBCUML_VERSION(CUML_VERSION_MAJOR, CUML_VERSION_MINOR) >= \
     CUML4R_LIBCUML_VERSION(24, 0))
__host__ double*& svmSupportData(ML::SVM::svmModel<double>& svm_model) {
  return svm_model.support_matrix.data;
}

__host__ double const* svmSupportData(
  ML::SVM::svmModel<double> const& svm_model) {
  return svm_model.support_matrix.data;
}
#else
__host__ double*& svmSupportData(ML::SVM::svmModel<double>& svm_model) {
  return svm_model.x_support;
}

__host__ double const* svmSupportData(
  ML::SVM::svmModel<double> const& svm_model) {
  return svm_model.x_support;
}
#endif

}  // namespace

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
  state[kSvmParamsVerbosity] = static_cast<int>(svm_params.verbosity);
  state[kSvmParamsEpsilon] = svm_params.epsilon;
  state[kSvmParamsType] = static_cast<int>(svm_params.svmType);

  return state;
}

__host__ Rcpp::List getState(ML::SVM::svmModel<double> const& svm_model,
                             raft::handle_t const& handle) {
  Rcpp::List state;
  cudaStream_t const stream = handle.get_stream();

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
    /*src=*/svmSupportData(svm_model),
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
#if (CUML4R_LIBCUML_VERSION(CUML_VERSION_MAJOR, CUML_VERSION_MINOR) >= \
     CUML4R_LIBCUML_VERSION(24, 0))
  svm_params.verbosity = static_cast<rapids_logger::level_enum>(
    Rcpp::as<int>(state[kSvmParamsVerbosity]));
#else
  svm_params.verbosity = state[kSvmParamsVerbosity];
#endif
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

  cudaStream_t const stream = handle.get_stream();

  CUDA_RT_CALL(cudaMalloc(&svm_model.dual_coefs, n_support * sizeof(double)));
  auto const h_dual_coefs =
    Rcpp::as<pinned_host_vector<double>>(state[kSvmModelDualCoefs]);
  CUDA_RT_CALL(cudaMemcpyAsync(
    /*dst=*/svm_model.dual_coefs,
    /*src=*/h_dual_coefs.data(),
    /*count=*/n_support * sizeof(double),
    /*kind=*/cudaMemcpyHostToDevice,
    /*stream=*/stream));

  CUDA_RT_CALL(
    cudaMalloc(&svmSupportData(svm_model), n_support * n_cols * sizeof(double)));
#if (CUML4R_LIBCUML_VERSION(CUML_VERSION_MAJOR, CUML_VERSION_MINOR) >= \
     CUML4R_LIBCUML_VERSION(24, 0))
  svm_model.support_matrix.nnz = n_support * n_cols;
  svm_model.support_matrix.indptr = nullptr;
  svm_model.support_matrix.indices = nullptr;
#endif
  auto const h_x_support =
    Rcpp::as<pinned_host_vector<double>>(state[kSvmModelSupportVectors]);
  CUDA_RT_CALL(cudaMemcpyAsync(
    /*dst=*/svmSupportData(svm_model),
    /*src=*/h_x_support.data(),
    /*count=*/n_support * n_cols * sizeof(double),
    /*kind=*/cudaMemcpyHostToDevice,
    /*stream=*/stream));

  CUDA_RT_CALL(cudaMalloc(&svm_model.support_idx, n_support * sizeof(int)));
  auto const h_support_idx =
    Rcpp::as<pinned_host_vector<int>>(state[kSvmModelSupportIdxes]);
  CUDA_RT_CALL(cudaMemcpyAsync(
    /*dst=*/svm_model.support_idx,
    /*src=*/h_support_idx.data(),
    /*count=*/n_support * sizeof(int),
    /*kind=*/cudaMemcpyHostToDevice,
    /*stream=*/stream));

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
    /*stream=*/stream));

  CUDA_RT_CALL(cudaStreamSynchronize(stream));
}

}  // namespace detail
}  // namespace cuml4r
