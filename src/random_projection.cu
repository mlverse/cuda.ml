#include "async_utils.cuh"
#include "cuda_utils.h"
#include "handle_utils.h"
#include "matrix_utils.h"
#include "pinned_host_vector.h"
#include "preprocessor.h"
#include "stream_allocator.h"

#include <cuml/random_projection/rproj_c.h>
#include <thrust/async/copy.h>
#include <thrust/device_vector.h>

#include <Rcpp.h>

#include <memory>
#include <vector>

namespace {

struct RPROJCtx {
  std::unique_ptr<raft::handle_t> const handle_;
  std::unique_ptr<ML::paramsRPROJ> const params_;
  std::unique_ptr<ML::rand_mat<double>> const randomMatrix_;

  __host__ RPROJCtx(
    std::unique_ptr<raft::handle_t> handle,
    std::unique_ptr<ML::paramsRPROJ> params,
    std::unique_ptr<ML::rand_mat<double>> random_matrix) noexcept
    : handle_(std::move(handle)),
      params_(std::move(params)),
      randomMatrix_(std::move(random_matrix)) {}
};

}  // namespace

namespace cuml4r {

__host__ size_t rproj_johnson_lindenstrauss_min_dim(size_t const n_samples,
                                                    double const eps) {
  return ML::johnson_lindenstrauss_min_dim(n_samples, eps);
}

__host__ SEXP rproj_fit(int const n_samples, int const n_features,
                        int const n_components, double const eps,
                        bool const gaussian_method, double const density,
                        int const random_state) {
  auto params = std::make_unique<ML::paramsRPROJ>();
  params->n_samples = n_samples;
  params->n_features = n_features;
  params->n_components = n_components;
  params->eps = eps;
  params->gaussian_method = gaussian_method;
  params->density = density;
  params->random_state = random_state;

  auto stream_view = cuml4r::stream_allocator::getOrCreateStream();
  auto handle = std::make_unique<raft::handle_t>();
  cuml4r::handle_utils::initializeHandle(*handle, stream_view.value());

  auto random_matrix = std::make_unique<ML::rand_mat<double>>(
    /*allocator=*/handle->get_device_allocator(),
    /*stream=*/stream_view.value());

  ML::RPROJfit(*handle, /*random_matrix=*/random_matrix.get(),
               /*params=*/params.get());

  auto rproj_ctx = std::make_unique<RPROJCtx>(
    /*handle=*/std::move(handle), /*params=*/std::move(params),
    /*random_matrix=*/std::move(random_matrix));

  return Rcpp::XPtr<RPROJCtx>(rproj_ctx.release());
}

__host__ Rcpp::NumericMatrix rproj_transform(SEXP rproj_ctx_xptr,
                                             Rcpp::NumericMatrix const& input) {
  auto rproj_ctx = Rcpp::XPtr<RPROJCtx>(rproj_ctx_xptr);
  auto const& handle = rproj_ctx->handle_;
  auto const stream = handle->get_stream();
  auto const& params = rproj_ctx->params_;
  auto const n_samples = params->n_samples;
  auto const n_components = params->n_components;

  // RPROJtransform input
  auto const m = cuml4r::Matrix<>(input, /*transpose=*/true);
  auto const& h_input = m.values;
  thrust::device_vector<double> d_input(h_input.size());
  auto CUML4R_ANONYMOUS_VARIABLE(input_h2d) = cuml4r::async_copy(
    stream, h_input.cbegin(), h_input.cend(), d_input.begin());

  // RPROJtransform output
  thrust::device_vector<double> d_output(n_samples * n_components);

  ML::RPROJtransform(/*handle=*/*handle, /*input=*/d_input.data().get(),
                     /*random_matrix=*/rproj_ctx->randomMatrix_.get(),
                     /*output=*/d_output.data().get(),
                     /*params=*/params.get());

  cuml4r::pinned_host_vector<double> h_output(d_output.size());
  auto CUML4R_ANONYMOUS_VARIABLE(output_d2h) = cuml4r::async_copy(
    stream, d_output.cbegin(), d_output.cend(), h_output.begin());

  CUDA_RT_CALL(cudaStreamSynchronize(stream));

  return Rcpp::transpose(
    Rcpp::NumericMatrix(n_components, n_samples, h_output.begin()));
}

}  // namespace cuml4r
