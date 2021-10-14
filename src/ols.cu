#include "async_utils.cuh"
#include "cuda_utils.h"
#include "handle_utils.h"
#include "matrix_utils.h"
#include "pinned_host_vector.h"
#include "preprocessor.h"
#include "stream_allocator.h"

#include <thrust/async/copy.h>
#include <thrust/device_vector.h>
#include <cuml/linear_model/glm.hpp>

#include <Rcpp.h>

#include <vector>

namespace cuml4r {
namespace {

constexpr auto kCoef = "coef";
constexpr auto kIntercept = "intercept";

}  // namespace

__host__ Rcpp::List ols_fit(Rcpp::NumericMatrix const& x,
                            Rcpp::NumericVector const& y,
                            bool const fit_intercept,
                            bool const normalize_input, int const algo) {
  auto const m = cuml4r::Matrix<>(x, /*transpose=*/true);
  auto const n_rows = m.numCols;
  auto const n_cols = m.numRows;

  auto stream_view = cuml4r::stream_allocator::getOrCreateStream();
  raft::handle_t handle;
  cuml4r::handle_utils::initializeHandle(handle, stream_view.value());

  // OLS input
  auto const& h_input = m.values;
  thrust::device_vector<double> d_input(h_input.size());
  auto CUML4R_ANONYMOUS_VARIABLE(input_h2d) = cuml4r::async_copy(
    stream_view.value(), h_input.cbegin(), h_input.cend(), d_input.begin());
  auto const h_labels = Rcpp::as<cuml4r::pinned_host_vector<double>>(y);
  thrust::device_vector<double> d_labels(h_labels.size());
  auto CUML4R_ANONYMOUS_VARIABLE(labels_h2d) = cuml4r::async_copy(
    stream_view.value(), h_labels.cbegin(), h_labels.cend(), d_labels.begin());

  // OLS output
  thrust::device_vector<double> d_coef(n_cols);
  thrust::device_vector<double> d_intercept(1);

  ML::GLM::olsFit(handle, /*input=*/d_input.data().get(), n_rows, n_cols,
                  /*labels=*/d_labels.data().get(),
                  /*coef=*/d_coef.data().get(),
                  /*intercept=*/d_intercept.data().get(), fit_intercept,
                  /*normalize=*/normalize_input, algo);

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  cuml4r::pinned_host_vector<double> h_coef(n_cols);
  cuml4r::pinned_host_vector<double> h_intercept(1);
  auto CUML4R_ANONYMOUS_VARIABLE(coef_d2h) = cuml4r::async_copy(
    stream_view.value(), d_coef.cbegin(), d_coef.cend(), h_coef.begin());
  auto CUML4R_ANONYMOUS_VARIABLE(intercept_d2h) =
    cuml4r::async_copy(stream_view.value(), d_intercept.cbegin(),
                       d_intercept.cend(), h_intercept.begin());

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  Rcpp::List model;
  model[kCoef] = Rcpp::NumericVector(h_coef.begin(), h_coef.end());
  model[kIntercept] = *h_intercept.begin();

  return model;
}

}  // namespace cuml4r
