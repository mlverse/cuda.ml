#include "async_utils.cuh"
#include "cuda_utils.h"
#include "handle_utils.h"
#include "matrix_utils.h"
#include "pinned_host_vector.h"
#include "preprocessor.h"
#include "qn_constants.h"
#include "stream_allocator.h"

#include <thrust/async/copy.h>
#include <thrust/device_vector.h>
#include <cuml/linear_model/glm.hpp>

#include <Rcpp.h>

#include <limits>

namespace cuml4r {

__host__ Rcpp::List qn_fit(Rcpp::NumericMatrix const& X,
                           Rcpp::IntegerVector const& y, int const n_classes,
                           int const loss_type, bool const fit_intercept,
                           double const l1, double const l2,
                           int const max_iters, double const tol,
                           double const delta, int const linesearch_max_iters,
                           int const lbfgs_memory,
                           Rcpp::NumericVector const& sample_weight) {
  auto const m = Matrix<>(X, /*transpose=*/true);
  auto const n_samples = m.numCols;
  auto const n_features = m.numRows;

  auto stream_view = stream_allocator::getOrCreateStream();
  auto handle = std::make_unique<raft::handle_t>();
  handle_utils::initializeHandle(*handle, stream_view.value());

  // QN input
  auto const& h_X = m.values;
  thrust::device_vector<double> d_X(h_X.size());
  auto CUML4R_ANONYMOUS_VARIABLE(X_h2d) =
    async_copy(stream_view.value(), h_X.cbegin(), h_X.cend(), d_X.begin());

  auto h_y(Rcpp::as<pinned_host_vector<double>>(y));
  thrust::device_vector<double> d_y(h_y.size());
  auto CUML4R_ANONYMOUS_VARIABLE(y_h2d) =
    async_copy(stream_view.value(), h_y.cbegin(), h_y.cend(), d_y.begin());

  thrust::device_vector<double> d_sample_weight;
  AsyncCopyCtx sample_weight_h2d;
  if (sample_weight.size() > 0) {
    d_sample_weight.resize(sample_weight.size());
    auto h_sample_weight(Rcpp::as<pinned_host_vector<double>>(sample_weight));
    sample_weight_h2d =
      async_copy(stream_view.value(), h_sample_weight.cbegin(),
                 h_sample_weight.cend(), d_sample_weight.begin());
  }

  auto const n_classes_dim = (n_classes - (loss_type == 0 ? 1 : 0));
  int const n_coefs_per_class = (fit_intercept ? n_features + 1 : n_features);
  thrust::device_vector<double> d_coefs(n_coefs_per_class * n_classes_dim);
  double objective = std::numeric_limits<double>::infinity();
  int n_iters = 0;

  ML::GLM::qnFit(
    /*handle=*/*handle, /*X=*/d_X.data().get(), /*X_col_major=*/true,
    /*y=*/d_y.data().get(), /*N=*/n_samples,
    /*D=*/n_features, /*C=*/n_classes, fit_intercept, l1, l2, max_iters,
    /*grad_tol=*/tol, /*change_tol=*/delta, linesearch_max_iters, lbfgs_memory,
    /*verbosity=*/0,
    /*w0=*/d_coefs.data().get(),
    /*f=*/&objective, /*num_iters=*/&n_iters, loss_type,
    /*sample_weight=*/d_sample_weight.empty() ? nullptr
                                              : d_sample_weight.data().get());

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  pinned_host_vector<double> h_coefs(d_coefs.size());
  auto CUML4R_ANONYMOUS_VARIABLE(coefs_d2h) = async_copy(
    stream_view.value(), d_coefs.cbegin(), d_coefs.cend(), h_coefs.begin());

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  Rcpp::List model;
  model[qn::kCoefs] =
    Rcpp::NumericMatrix(n_coefs_per_class, n_classes_dim, h_coefs.cbegin());
  model[qn::kFitIntercept] = fit_intercept;
  model[qn::kLossType] = loss_type;
  model[qn::kNumClasses] = n_classes;
  model[qn::kObjective] = objective;
  model[qn::kNumIters] = n_iters;

  return model;
}

Rcpp::NumericVector qn_predict(Rcpp::NumericMatrix const& X,
                               int const n_classes,
                               Rcpp::NumericMatrix const& coefs,
                               int const loss_type, bool const fit_intercept) {
  auto const m_X = Matrix<>(X, /*transpose=*/true);
  auto const n_samples = m_X.numCols;
  auto const n_features = m_X.numRows;

  auto stream_view = stream_allocator::getOrCreateStream();
  auto handle = std::make_unique<raft::handle_t>();
  handle_utils::initializeHandle(*handle, stream_view.value());

  // QN input
  auto const& h_X = m_X.values;
  thrust::device_vector<double> d_X(h_X.size());
  auto CUML4R_ANONYMOUS_VARIABLE(X_h2d) =
    async_copy(stream_view.value(), h_X.cbegin(), h_X.cend(), d_X.begin());

  auto const m_coefs = Matrix<>(coefs, /*transpose=*/true);
  auto const& h_coefs = m_coefs.values;
  thrust::device_vector<double> d_coefs(h_coefs.size());
  auto CUML4R_ANONYMOUS_VARIABLE(coefs_h2d) = async_copy(
    stream_view.value(), h_coefs.cbegin(), h_coefs.cend(), d_coefs.begin());

  // QN output
  thrust::device_vector<double> d_preds(n_samples);

  ML::GLM::qnPredict(
    /*cuml_handle=*/*handle,
    /*X=*/d_X.data().get(),
    /*X_col_major=*/true,
    /*N=*/n_samples,
    /*D=*/n_features,
    /*C=*/n_classes, fit_intercept,
    /*params=*/d_coefs.data().get(), loss_type,
    /*preds=*/d_preds.data().get());

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  pinned_host_vector<double> h_preds(n_samples);
  auto CUML4R_ANONYMOUS_VARIABLE(preds_d2h) = async_copy(
    stream_view.value(), d_preds.cbegin(), d_preds.cend(), h_preds.begin());

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  return Rcpp::NumericVector(h_preds.begin(), h_preds.end());
}

}  // namespace cuml4r
