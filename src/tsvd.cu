#include "async_utils.cuh"
#include "cuda_utils.h"
#include "handle_utils.h"
#include "matrix_utils.h"
#include "pinned_host_vector.h"
#include "preprocessor.h"
#include "stream_allocator.h"

#include <thrust/async/copy.h>
#include <thrust/device_vector.h>
#include <cuml/decomposition/tsvd.hpp>

#include <Rcpp.h>

#include <memory>

namespace cuml4r {

__host__ Rcpp::List tsvd_fit_transform(Rcpp::NumericMatrix const& x,
                                       double const tol, int const n_iters,
                                       int const verbosity,
                                       int const n_components, int const algo,
                                       bool const transform_input) {
  Rcpp::List result;

  auto const m = Matrix<>(x, /*transpose=*/true);
  auto const n_rows = m.numCols;
  auto const n_cols = m.numRows;

  auto params = std::make_unique<ML::paramsTSVD>();
  params->n_rows = n_rows;
  params->n_cols = n_cols;
  params->gpu_id = currentDevice();
  params->tol = tol;
  params->n_iterations = n_iters;
  params->verbose = verbosity;
  params->n_components = n_components;
  params->algorithm = static_cast<ML::solver>(algo);

  auto stream_view = stream_allocator::getOrCreateStream();
  raft::handle_t handle;
  handle_utils::initializeHandle(handle, stream_view.value());

  // TSVD input
  auto const& h_input = m.values;
  thrust::device_vector<double> d_input(h_input.size());
  auto CUML4R_ANONYMOUS_VARIABLE(input_h2d) = async_copy(
    stream_view.value(), h_input.cbegin(), h_input.cend(), d_input.begin());

  // TSVD outputs
  thrust::device_vector<double> d_transformed_data;
  if (transform_input) {
    d_transformed_data.resize(n_rows * n_components);
  }
  thrust::device_vector<double> d_components(n_cols * n_components);
  thrust::device_vector<double> d_singular_vals(n_components);
  thrust::device_vector<double> d_explained_var;
  if (transform_input) {
    d_explained_var.resize(n_components);
  }
  thrust::device_vector<double> d_explained_var_ratio;
  if (transform_input) {
    d_explained_var_ratio.resize(n_components);
  }

  if (transform_input) {
    ML::tsvdFitTransform(
      handle,
      /*input=*/d_input.data().get(),
      /*trans_input=*/d_transformed_data.data().get(),
      /*components=*/d_components.data().get(),
      /*explained_var=*/d_explained_var.data().get(),
      /*explained_var_ratio=*/d_explained_var_ratio.data().get(),
      /*singular_vals=*/d_singular_vals.data().get(),
      /*prms=*/*params);
  } else {
    ML::tsvdFit(handle,
                /*input=*/d_input.data().get(),
                /*components=*/d_components.data().get(),
                /*singular_vals=*/d_singular_vals.data().get(),
                /*prms=*/*params);
  }

  pinned_host_vector<double> h_transformed_data;
  if (transform_input) {
    h_transformed_data.resize(d_transformed_data.size());
  }
  pinned_host_vector<double> h_components(n_cols * n_components);
  pinned_host_vector<double> h_explained_var;
  if (transform_input) {
    h_explained_var.resize(d_explained_var.size());
  }
  pinned_host_vector<double> h_explained_var_ratio;
  if (transform_input) {
    h_explained_var_ratio.resize(d_explained_var_ratio.size());
  }
  pinned_host_vector<double> h_singular_vals(n_components);

  AsyncCopyCtx transformed_data_d2h;
  if (transform_input) {
    transformed_data_d2h =
      async_copy(stream_view.value(), d_transformed_data.cbegin(),
                 d_transformed_data.cend(), h_transformed_data.begin());
  }
  auto CUML4R_ANONYMOUS_VARIABLE(components_d2h) =
    async_copy(stream_view.value(), d_components.cbegin(), d_components.cend(),
               h_components.begin());
  AsyncCopyCtx explained_var_d2h;
  if (transform_input) {
    explained_var_d2h =
      async_copy(stream_view.value(), d_explained_var.cbegin(),
                 d_explained_var.cend(), h_explained_var.begin());
  }
  AsyncCopyCtx explained_var_ratio_d2h;
  if (transform_input) {
    explained_var_ratio_d2h =
      async_copy(stream_view.value(), d_explained_var_ratio.cbegin(),
                 d_explained_var_ratio.cend(), h_explained_var_ratio.begin());
  }
  auto CUML4R_ANONYMOUS_VARIABLE(singular_vals_d2h) =
    async_copy(stream_view.value(), d_singular_vals.cbegin(),
               d_singular_vals.cend(), h_singular_vals.begin());

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  result["components"] =
    Rcpp::NumericMatrix(n_components, n_cols, h_components.begin());
  if (transform_input) {
    result["explained_variance"] =
      Rcpp::NumericVector(h_explained_var.begin(), h_explained_var.end());
    result["explained_variance_ratio"] = Rcpp::NumericVector(
      h_explained_var_ratio.begin(), h_explained_var_ratio.end());
  }
  result["singular_values"] =
    Rcpp::NumericVector(h_singular_vals.begin(), h_singular_vals.end());
  if (transform_input) {
    result["transformed_data"] =
      Rcpp::NumericMatrix(n_rows, n_components, h_transformed_data.begin());
  }
  result["tsvd_params"] = Rcpp::XPtr<ML::paramsTSVD>(params.release());

  return result;
}

__host__ Rcpp::NumericMatrix tsvd_transform(Rcpp::List model,
                                            Rcpp::NumericMatrix const& x) {
  auto const m = Matrix<>(x, /*transpose=*/true);
  auto const& h_input = m.values;
  auto const components = Matrix<>(model["components"], /*transpose=*/true);
  auto const& h_components = components.values;
  Rcpp::XPtr<ML::paramsTSVD> const params = model["tsvd_params"];

  auto stream_view = stream_allocator::getOrCreateStream();
  raft::handle_t handle;
  handle_utils::initializeHandle(handle, stream_view.value());

  // transform inputs
  thrust::device_vector<double> d_input(h_input.size());
  thrust::device_vector<double> d_components(h_components.size());

  auto CUML4R_ANONYMOUS_VARIABLE(input_h2d) = async_copy(
    stream_view.value(), h_input.cbegin(), h_input.cend(), d_input.begin());
  auto CUML4R_ANONYMOUS_VARIABLE(components_h2d) =
    async_copy(stream_view.value(), h_components.cbegin(), h_components.cend(),
               d_components.begin());

  // transform output
  thrust::device_vector<double> d_result(params->n_rows * params->n_components);

  ML::tsvdTransform(handle, /*input=*/d_input.data().get(),
                    /*components=*/d_components.data().get(),
                    /*trans_input=*/d_result.data().get(),
                    /*prms=*/*params);

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  pinned_host_vector<double> h_result(d_result.size());
  auto CUML4R_ANONYMOUS_VARIABLE(result_d2h) = async_copy(
    stream_view.value(), d_result.cbegin(), d_result.cend(), h_result.begin());

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  return Rcpp::NumericMatrix(params->n_rows, params->n_components,
                             h_result.begin());
}

__host__ Rcpp::NumericMatrix tsvd_inverse_transform(
  Rcpp::List model, Rcpp::NumericMatrix const& x) {
  auto const m = Matrix<>(x, /*transpose=*/true);
  auto const& h_input = m.values;
  auto const components = Matrix<>(model["components"], /*transpose=*/true);
  auto const& h_components = components.values;
  Rcpp::XPtr<ML::paramsTSVD> params = model["tsvd_params"];

  auto stream_view = stream_allocator::getOrCreateStream();
  raft::handle_t handle;
  handle_utils::initializeHandle(handle, stream_view.value());

  // inverse transform inputs
  thrust::device_vector<double> d_input(h_input.size());
  thrust::device_vector<double> d_components(h_components.size());

  auto CUML4R_ANONYMOUS_VARIABLE(input_h2d) = async_copy(
    stream_view.value(), h_input.cbegin(), h_input.cend(), d_input.begin());
  auto CUML4R_ANONYMOUS_VARIABLE(components_h2d) =
    async_copy(stream_view.value(), h_components.cbegin(), h_components.cend(),
               d_components.begin());

  // inverse transform output
  thrust::device_vector<double> d_result(params->n_rows * params->n_cols);

  ML::tsvdInverseTransform(handle, /*trans_input=*/d_input.data().get(),
                           /*componenets=*/d_components.data().get(),
                           /*input=*/d_result.data().get(),
                           /*prms=*/*params);

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  pinned_host_vector<double> h_result(d_result.size());
  auto CUML4R_ANONYMOUS_VARIABLE(result_d2h) = async_copy(
    stream_view.value(), d_result.cbegin(), d_result.cend(), h_result.begin());

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  return Rcpp::NumericMatrix(params->n_rows, params->n_cols, h_result.begin());
}

}  // namespace cuml4r
