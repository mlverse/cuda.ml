#include "async_utils.cuh"
#include "cuda_utils.h"
#include "handle_utils.h"
#include "matrix_utils.h"
#include "pinned_host_vector.h"
#include "preprocessor.h"
#include "stream_allocator.h"

#include <cuml/manifold/tsne.h>
#include <thrust/async/copy.h>
#include <thrust/device_vector.h>
#include <cuml/manifold/umap.hpp>

#include <Rcpp.h>

namespace cuml4r {

__host__ Rcpp::NumericMatrix tsne_fit(
  Rcpp::NumericMatrix const& x, int const dim, int const n_neighbors,
  float const theta, float const epssq, float const perplexity,
  int const perplexity_max_iter, float const perplexity_tol,
  float const early_exaggeration, float const late_exaggeration,
  int const exaggeration_iter, float const min_gain,
  float const pre_learning_rate, float const post_learning_rate,
  int const max_iter, float const min_grad_norm, float const pre_momentum,
  float const post_momentum, long long const random_state, int const verbosity,
  bool const initialize_embeddings, bool const square_distances,
  int const algo) {
  auto const m_x = Matrix<float>(x, /*transpose=*/false);
  auto const n_samples = m_x.numRows;
  auto const n_features = m_x.numCols;

  ML::TSNEParams params;
  params.dim = dim;
  params.n_neighbors = n_neighbors;
  params.theta = theta;
  params.epssq = epssq;
  params.perplexity = perplexity;
  params.perplexity_max_iter = perplexity_max_iter;
  params.perplexity_tol = perplexity_tol;
  params.early_exaggeration = early_exaggeration;
  params.late_exaggeration = late_exaggeration;
  params.exaggeration_iter = exaggeration_iter;
  params.min_gain = min_gain;
  params.pre_learning_rate = pre_learning_rate;
  params.post_learning_rate = post_learning_rate;
  params.max_iter = max_iter;
  params.min_grad_norm = min_grad_norm;
  params.pre_momentum = pre_momentum;
  params.post_momentum = post_momentum;
  params.random_state = random_state;
  params.verbosity = verbosity;
  params.initialize_embeddings = initialize_embeddings;
  params.square_distances = square_distances;
  params.algorithm = static_cast<ML::TSNE_ALGORITHM>(algo);

  auto stream_view = stream_allocator::getOrCreateStream();
  raft::handle_t handle;
  handle_utils::initializeHandle(handle, stream_view.value());

  // TSNE input
  auto const& h_x = m_x.values;
  thrust::device_vector<float> d_x(h_x.size());
  auto CUML4R_ANONYMOUS_VARIABLE(x_h2d) =
    async_copy(stream_view.value(), h_x.cbegin(), h_x.cend(), d_x.begin());

  // TSNE output
  thrust::device_vector<float> d_y(n_samples * dim);

  ML::TSNE_fit(
    handle, /*X=*/d_x.data().get(), /*Y=*/d_y.data().get(), /*n=*/n_samples,
    /*p=*/n_features, /*knn_indices=*/nullptr, /*knn_dists=*/nullptr, params);

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  pinned_host_vector<float> h_y(d_y.size());
  auto CUML4R_ANONYMOUS_VARIABLE(y_d2h) =
    async_copy(stream_view.value(), d_y.cbegin(), d_y.cend(), h_y.begin());

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  return Rcpp::NumericMatrix(n_samples, dim, h_y.begin());
}

}  // namespace cuml4r
