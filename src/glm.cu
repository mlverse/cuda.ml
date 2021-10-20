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

namespace cuml4r {

Rcpp::NumericVector glm_predict(Rcpp::NumericMatrix const& input,
                                Rcpp::NumericVector const& coef,
                                double const intercept) {
  auto const m = Matrix<>(input, /*transpose=*/true);
  auto const n_rows = m.numCols;
  auto const n_cols = m.numRows;

  auto stream_view = stream_allocator::getOrCreateStream();
  raft::handle_t handle;
  handle_utils::initializeHandle(handle, stream_view.value());

  // GLM input
  auto const& h_input = m.values;
  thrust::device_vector<double> d_input(h_input.size());
  auto CUML4R_ANONYMOUS_VARIABLE(input_h2d) = async_copy(
    stream_view.value(), h_input.cbegin(), h_input.cend(), d_input.begin());
  // GLM coefficients & intercept
  auto const h_coef = Rcpp::as<pinned_host_vector<double>>(coef);
  thrust::device_vector<double> d_coef(h_coef.size());
  auto CUML4R_ANONYMOUS_VARIABLE(coef_h2d) = async_copy(
    stream_view.value(), h_coef.cbegin(), h_coef.cend(), d_coef.begin());

  // GLM output
  thrust::device_vector<double> d_preds(n_rows);
  ML::GLM::gemmPredict(handle, /*input=*/d_input.data().get(), n_rows, n_cols,
                       /*coef=*/d_coef.data().get(), intercept,
                       /*preds=*/d_preds.data().get());

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  pinned_host_vector<double> h_preds(n_rows);
  auto CUML4R_ANONYMOUS_VARIABLE(preds_d2h) = async_copy(
    stream_view.value(), d_preds.cbegin(), d_preds.cend(), h_preds.begin());

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  return Rcpp::NumericVector(h_preds.begin(), h_preds.end());
}

}  // namespace cuml4r
