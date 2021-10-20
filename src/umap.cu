#include "async_utils.cuh"
#include "cuda_utils.h"
#include "handle_utils.h"
#include "matrix_utils.h"
#include "pinned_host_vector.h"
#include "preprocessor.h"
#include "stream_allocator.h"

#include <cuml/manifold/umapparams.h>
#include <thrust/async/copy.h>
#include <thrust/device_vector.h>
#include <cuml/manifold/umap.hpp>

#include <Rcpp.h>

#include <cmath>
#include <memory>
#include <type_traits>

namespace {

/*
 * The 'ML::UMAPParams::target_weights' parameter was renamed to 'target_weight'
 * in RAPIDS cuML v21.06 or above, so, using SFINAE here to be compatible with
 * both versions of the 'ML::UMAPParams' definitions.
 */

// for cuML v21.06 or above
template <typename T>
__host__ void set_target_weight(
  T& params,
  typename std::remove_reference<decltype(T::target_weight)>::type const
    w) noexcept {
  params.target_weight = w;
}

// for earlier versions of cuML
template <typename T>
__host__ void set_target_weight(
  T& params,
  typename std::remove_reference<decltype(T::target_weights)>::type const
    w) noexcept {
  params.target_weights = w;
}

}  // namespace

namespace cuml4r {

__host__ Rcpp::List umap_fit(
  Rcpp::NumericMatrix const& x, Rcpp::NumericVector const& y,
  int const n_neighbors, int const n_components, int const n_epochs,
  float const learning_rate, float const min_dist, float const spread,
  float const set_op_mix_ratio, int const local_connectivity,
  float const repulsion_strength, int const negative_sample_rate,
  float const transform_queue_size, int const verbosity, float const a,
  float const b, int const init, int const target_n_neighbors,
  int const target_metric, float const target_weight,
  uint64_t const random_state, bool const deterministic) {
  Rcpp::List model;

  auto const m_x = Matrix<float>(x, /*transpose=*/false);
  auto const n_samples = m_x.numRows;
  auto const n_features = m_x.numCols;

  auto stream_view = stream_allocator::getOrCreateStream();
  raft::handle_t handle;
  handle_utils::initializeHandle(handle, stream_view.value());

  auto params = std::make_unique<ML::UMAPParams>();
  params->n_neighbors = n_neighbors;
  params->n_components = n_components;
  params->n_epochs = n_epochs;
  params->learning_rate = learning_rate;
  params->min_dist = min_dist;
  params->spread = spread;
  params->set_op_mix_ratio = set_op_mix_ratio;
  params->local_connectivity = local_connectivity;
  params->repulsion_strength = repulsion_strength;
  params->negative_sample_rate = negative_sample_rate;
  params->transform_queue_size = transform_queue_size;
  params->verbosity = verbosity;
  if (std::isnan(a) || std::isnan(b)) {
    ML::UMAP::find_ab(handle, params.get());
  } else {
    params->a = a;
    params->b = b;
  }
  params->init = init;
  params->target_n_neighbors = target_n_neighbors;
  params->target_metric =
    static_cast<ML::UMAPParams::MetricType>(target_metric);
  set_target_weight(*params, target_weight);
  params->random_state = random_state;
  params->deterministic = deterministic;

  // UMAP input
  auto const& h_x = m_x.values;
  thrust::device_vector<float> d_x(h_x.size());
  auto CUML4R_ANONYMOUS_VARIABLE(x_h2d) =
    async_copy(stream_view.value(), h_x.cbegin(), h_x.cend(), d_x.begin());
  thrust::device_vector<float> d_y;
  AsyncCopyCtx y_h2d;
  if (y.size() > 0) {
    auto const h_y = Rcpp::as<pinned_host_vector<float>>(y);
    d_y.resize(y.size());
    y_h2d =
      async_copy(stream_view.value(), h_y.cbegin(), h_y.cend(), d_y.begin());
  }

  // UMAP output
  thrust::device_vector<float> d_embedding(n_samples * n_components);

  ML::UMAP::fit(handle, /*X=*/d_x.data().get(),
                /*y=*/(y.size() > 0 ? d_y.data().get() : nullptr),
                /*n=*/n_samples,
                /*d=*/n_features,
                /*knn_indices=*/nullptr,
                /*knn_dists=*/nullptr,
                /*params=*/params.get(),
                /*embeddings=*/d_embedding.data().get());

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  pinned_host_vector<float> h_embedding(d_embedding.size());
  auto CUML4R_ANONYMOUS_VARIABLE(embedding_d2h) =
    async_copy(stream_view.value(), d_embedding.cbegin(), d_embedding.cend(),
               h_embedding.begin());

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  model["umap_params"] = Rcpp::XPtr<ML::UMAPParams>(params.release());
  model["embedding"] = Rcpp::transpose(
    Rcpp::NumericMatrix(n_components, n_samples, h_embedding.begin()));
  model["n_samples"] = n_samples;
  model["x"] = x;

  return model;
}

__host__ Rcpp::NumericMatrix umap_transform(Rcpp::List const& model,
                                            Rcpp::NumericMatrix const& x) {
  auto const m_x = Matrix<float>(x, /*transpose=*/false);
  auto const n_samples = m_x.numRows;
  auto const n_features = m_x.numCols;
  auto const m_orig = Matrix<float>(model["x"], /*transpose=*/false);
  auto const m_embedding =
    Matrix<float>(model["embedding"], /*transpose=*/false);
  Rcpp::XPtr<ML::UMAPParams> params = model["umap_params"];

  auto stream_view = stream_allocator::getOrCreateStream();
  raft::handle_t handle;
  handle_utils::initializeHandle(handle, stream_view.value());

  // UMAP transform input
  auto const& h_x = m_x.values;
  thrust::device_vector<float> d_x(h_x.size());
  auto CUML4R_ANONYMOUS_VARIABLE(x_h2d) =
    async_copy(stream_view.value(), h_x.cbegin(), h_x.cend(), d_x.begin());
  auto const& h_orig_x = m_orig.values;
  thrust::device_vector<float> d_orig_x(h_orig_x.size());
  auto CUML4R_ANONYMOUS_VARIABLE(orig_x_h2d) = async_copy(
    stream_view.value(), h_orig_x.cbegin(), h_orig_x.cend(), d_orig_x.begin());
  auto const& h_embedding = m_embedding.values;
  thrust::device_vector<float> d_embedding(h_embedding.size());
  auto CUML4R_ANONYMOUS_VARIABLE(orig_x_h2d) =
    async_copy(stream_view.value(), h_embedding.cbegin(), h_embedding.cend(),
               d_embedding.begin());

  // UMAP transform output
  thrust::device_vector<float> d_transformed(n_samples * m_embedding.numCols);

  ML::UMAP::transform(
    handle, /*X=*/d_x.data().get(), /*n=*/n_samples, /*d=*/n_features,
    /*knn_indices=*/nullptr, /*knn_dists=*/nullptr,
    /*orig_x=*/d_orig_x.data().get(),
    /*orig_n=*/m_orig.numRows, /*embedding=*/d_embedding.data().get(),
    /*embedding_n=*/m_embedding.numRows,
    /*params=*/params.get(), /*transformed=*/d_transformed.data().get());

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  pinned_host_vector<float> h_transformed(d_transformed.size());
  auto CUML4R_ANONYMOUS_VARIABLE(transformed_d2h) =
    async_copy(stream_view.value(), d_transformed.cbegin(),
               d_transformed.cend(), h_transformed.begin());

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  return Rcpp::transpose(
    Rcpp::NumericMatrix(m_embedding.numCols, n_samples, h_transformed.begin()));
}

__host__ Rcpp::List umap_get_state(Rcpp::List const& model) {
  Rcpp::List state;

  {
    Rcpp::List umap_params_list;
    Rcpp::XPtr<ML::UMAPParams const> const umap_params = model["umap_params"];

    umap_params_list["n_neighbors"] = umap_params->n_neighbors;
    umap_params_list["n_components"] = umap_params->n_components;
    umap_params_list["n_epochs"] = umap_params->n_epochs;
    umap_params_list["learning_rate"] = umap_params->learning_rate;
    umap_params_list["min_dist"] = umap_params->min_dist;
    umap_params_list["spread"] = umap_params->spread;
    umap_params_list["set_op_mix_ratio"] = umap_params->set_op_mix_ratio;
    umap_params_list["local_connectivity"] = umap_params->local_connectivity;
    umap_params_list["repulsion_strength"] = umap_params->repulsion_strength;
    umap_params_list["negative_sample_rate"] =
      umap_params->negative_sample_rate;
    umap_params_list["transform_queue_size"] =
      umap_params->transform_queue_size;
    umap_params_list["verbosity"] = umap_params->verbosity;
    umap_params_list["a"] = umap_params->a;
    umap_params_list["b"] = umap_params->b;
    umap_params_list["init"] = umap_params->init;
    umap_params_list["target_n_neighbors"] = umap_params->target_n_neighbors;
    umap_params_list["target_metric"] =
      static_cast<int>(umap_params->target_metric);
    umap_params_list["target_weight"] =
      static_cast<int>(umap_params->target_weight);
    umap_params_list["random_state"] = umap_params->random_state;
    umap_params_list["deterministic"] = umap_params->deterministic;
    state["umap_params"] = std::move(umap_params_list);
  }
  state["embedding"] = model["embedding"];
  state["n_samples"] = model["n_samples"];
  state["x"] = model["x"];

  return state;
}

__host__ Rcpp::List umap_set_state(Rcpp::List const& state) {
  Rcpp::List model;

  {
    auto umap_params = std::make_unique<ML::UMAPParams>();
    Rcpp::List const& umap_params_list = state["umap_params"];

    umap_params->n_neighbors = umap_params_list["n_neighbors"];
    umap_params->n_components = umap_params_list["n_components"];
    umap_params->n_epochs = umap_params_list["n_epochs"];
    umap_params->learning_rate = umap_params_list["learning_rate"];
    umap_params->min_dist = umap_params_list["min_dist"];
    umap_params->spread = umap_params_list["spread"];
    umap_params->set_op_mix_ratio = umap_params_list["set_op_mix_ratio"];
    umap_params->local_connectivity = umap_params_list["local_connectivity"];
    umap_params->repulsion_strength = umap_params_list["repulsion_strength"];
    umap_params->negative_sample_rate =
      umap_params_list["negative_sample_rate"];
    umap_params->transform_queue_size =
      umap_params_list["transform_queue_size"];
    umap_params->verbosity = umap_params_list["verbosity"];
    umap_params->a = umap_params_list["a"];
    umap_params->b = umap_params_list["b"];
    umap_params->init = umap_params_list["init"];
    umap_params->target_n_neighbors = umap_params_list["target_n_neighbors"];
    umap_params->target_metric = static_cast<ML::UMAPParams::MetricType>(
      Rcpp::as<int>(umap_params_list["target_metric"]));
    umap_params->target_weight = umap_params_list["target_weight"];
    umap_params->random_state = umap_params_list["random_state"];
    umap_params->deterministic = umap_params_list["deterministic"];
    model["umap_params"] = Rcpp::XPtr<ML::UMAPParams>(umap_params.release());
  }
  model["embedding"] = state["embedding"];
  model["n_samples"] = state["n_samples"];
  model["x"] = state["x"];

  return model;
}

}  // namespace cuml4r
