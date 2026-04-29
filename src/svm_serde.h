#pragma once

#include "preprocessor.h"

#include <cuml/matrix/kernelparams.h>
#include <cuml/svm/svm_model.h>
#include <cuml/svm/svm_parameter.h>
#include <cuml/version_config.hpp>

#include <Rcpp.h>

#if (CUML4R_LIBCUML_VERSION(CUML_VERSION_MAJOR, CUML_VERSION_MINOR) >= \
     CUML4R_LIBCUML_VERSION(21, 10))

namespace ML {
namespace SVM {

using svmParameter = SvmParameter;

template <typename math_t>
using svmModel = SvmModel<math_t>;

}  // namespace SVM
}  // namespace ML

#endif

namespace cuml4r {
namespace detail {

constexpr auto kKernelParamsType = "type";
constexpr auto kKernelParamsDegree = "degree";
constexpr auto kKernelParamsGamma = "gamma";
constexpr auto kKernelParamsCoef0 = "coef0";
constexpr auto kSvmParamsC = "C";
constexpr auto kSvmParamsCacheSize = "cache_size";
constexpr auto kSvmParamsMaxIter = "max_iter";
constexpr auto kSvmParamsNoChangeSteps = "nochange_steps";
constexpr auto kSvmParamsTol = "tol";
constexpr auto kSvmParamsVerbosity = "verbosity";
constexpr auto kSvmParamsEpsilon = "epsilon";
constexpr auto kSvmParamsType = "type";
constexpr auto kSvmModelNumSupportVectors = "n_support";
constexpr auto kSvmModelNumCols = "n_cols";
constexpr auto kSvmModelB = "b";
constexpr auto kSvmModelDualCoefs = "dual_coefs";
constexpr auto kSvmModelSupportVectors = "x_support";
constexpr auto kSvmModelSupportIdxes = "support_idx";
constexpr auto kSvmModelNumClasses = "n_classes";
constexpr auto kSvmModelUniqueLabels = "unique_labels";

Rcpp::List getState(MLCommon::Matrix::KernelParams const& kernel_params);

Rcpp::List getState(ML::SVM::svmParameter const& svm_params);

Rcpp::List getState(ML::SVM::svmModel<double> const& svm_model,
                    raft::handle_t const& handle);

void setState(MLCommon::Matrix::KernelParams& kernel_params,
              Rcpp::List const& state);

void setState(ML::SVM::svmParameter& svm_params, Rcpp::List const& state);

void setState(ML::SVM::svmModel<double>& svm_model,
              raft::handle_t const& handle, Rcpp::List const& state);

}  // namespace detail
}  // namespace cuml4r
