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

constexpr auto kRProjParamsNumSamples = "n_samples";
constexpr auto kRProjParamsNumFeatures = "n_features";
constexpr auto kRProjParamsNumComponents = "n_components";
constexpr auto kRProjParamsEps = "eps";
constexpr auto kRProjParamsGaussianMethod = "gaussian_method";
constexpr auto kRProjParamsDensity = "density";
constexpr auto kRProjParamsDenseOutput = "dense_output";
constexpr auto kRProjParamsRandomState = "random_state";

constexpr auto kRandMatDenseData = "dense_data";
constexpr auto kRandMatIndices = "indices";
constexpr auto kRandMatIndPtr = "indptr";
constexpr auto kRandMatSparseData = "sparse_data";
constexpr auto kRandMatType = "type";

template <typename T>
__host__ Rcpp::Vector<Rcpp::traits::r_sexptype_traits<T>::rtype> toRcppVector(
  MLCommon::device_buffer<T> const& d_data) {
  auto const stream = d_data.get_stream();

  cuml4r::pinned_host_vector<T> h_data(d_data.size());
  CUDA_RT_CALL(cudaMemcpyAsync(/*dst=*/h_data.data(), /*src=*/d_data.data(),
                               /*count=*/sizeof(T) * d_data.size(),
                               /*kind=*/cudaMemcpyDeviceToHost, stream));
  CUDA_RT_CALL(cudaStreamSynchronize(stream));

  return Rcpp::Vector<Rcpp::traits::r_sexptype_traits<T>::rtype>(h_data.begin(),
                                                                 h_data.end());
}

template <typename T>
__host__ void toDeviceBuffer(
  MLCommon::device_buffer<T>& dst,
  Rcpp::Vector<Rcpp::traits::r_sexptype_traits<T>::rtype> const& src) {
  auto const stream = dst.get_stream();

  auto h_data = Rcpp::as<cuml4r::pinned_host_vector<T>>(src);
  dst.resize(src.size());
  CUDA_RT_CALL(cudaMemcpyAsync(/*dst=*/dst.data(), /*src=*/h_data.data(),
                               /*count=*/sizeof(T) * h_data.size(),
                               /*kind=*/cudaMemcpyHostToDevice, stream));
  CUDA_RT_CALL(cudaStreamSynchronize(stream));
}

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

  __host__ RPROJCtx(std::unique_ptr<raft::handle_t> handle,
                    Rcpp::List const& model_state)
    : handle_(std::move(handle)),
      params_(std::make_unique<ML::paramsRPROJ>()),
      randomMatrix_(std::make_unique<ML::rand_mat<double>>(
        handle_->get_device_allocator(), handle_->get_stream())) {
    setState(model_state);
  }

  __host__ Rcpp::List getState() const {
    Rcpp::List model_state;

    model_state[kRProjParamsNumSamples] = params_->n_samples;
    model_state[kRProjParamsNumFeatures] = params_->n_features;
    model_state[kRProjParamsNumComponents] = params_->n_components;
    model_state[kRProjParamsEps] = params_->eps;
    model_state[kRProjParamsGaussianMethod] = params_->gaussian_method;
    model_state[kRProjParamsDensity] = params_->density;
    model_state[kRProjParamsDenseOutput] = params_->dense_output;
    model_state[kRProjParamsRandomState] = params_->random_state;

    model_state[kRandMatDenseData] = toRcppVector(randomMatrix_->dense_data);
    model_state[kRandMatIndices] = toRcppVector(randomMatrix_->indices);
    model_state[kRandMatIndPtr] = toRcppVector(randomMatrix_->indptr);
    model_state[kRandMatSparseData] = toRcppVector(randomMatrix_->sparse_data);
    model_state[kRandMatType] = static_cast<int>(randomMatrix_->type);

    return model_state;
  }

  __host__ void setState(Rcpp::List const& model_state) {
    params_->n_samples = model_state[kRProjParamsNumSamples];
    params_->n_features = model_state[kRProjParamsNumFeatures];
    params_->n_components = model_state[kRProjParamsNumComponents];
    params_->eps = model_state[kRProjParamsEps];
    params_->gaussian_method = model_state[kRProjParamsGaussianMethod];
    params_->density = model_state[kRProjParamsDensity];
    params_->dense_output = model_state[kRProjParamsDenseOutput];
    params_->random_state = model_state[kRProjParamsRandomState];

    toDeviceBuffer(
      randomMatrix_->dense_data,
      Rcpp::as<Rcpp::NumericVector>(model_state[kRandMatDenseData]));
    toDeviceBuffer(randomMatrix_->indices,
                   Rcpp::as<Rcpp::IntegerVector>(model_state[kRandMatIndices]));
    toDeviceBuffer(randomMatrix_->indptr,
                   Rcpp::as<Rcpp::IntegerVector>(model_state[kRandMatIndPtr]));
    toDeviceBuffer(
      randomMatrix_->sparse_data,
      Rcpp::as<Rcpp::NumericVector>(model_state[kRandMatSparseData]));
    randomMatrix_->type = static_cast<ML::random_matrix_type>(
      Rcpp::as<int>(model_state[kRandMatType]));
  }
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

__host__ Rcpp::List rproj_get_state(SEXP model) {
  return Rcpp::XPtr<RPROJCtx>(model)->getState();
}

__host__ SEXP rproj_set_state(Rcpp::List const& model_state) {
  auto stream_view = cuml4r::stream_allocator::getOrCreateStream();
  auto handle = std::make_unique<raft::handle_t>();
  cuml4r::handle_utils::initializeHandle(*handle, stream_view.value());
  auto model =
    std::make_unique<RPROJCtx>(/*handle=*/std::move(handle), model_state);

  return Rcpp::XPtr<RPROJCtx>(model.release());
}

}  // namespace cuml4r
