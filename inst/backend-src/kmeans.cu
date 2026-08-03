#include "async_utils.cuh"
#include "cuda_utils.h"
#include "handle_utils.h"
#include "matrix_utils.h"
#include "pinned_host_vector.h"
#include "preprocessor.h"
#include "stream_allocator.h"

#include <cuml/cluster/kmeans.hpp>
#include <thrust/device_vector.h>

#include <Rcpp.h>

#include <memory>

namespace cuml4r {

__host__ Rcpp::List kmeans(Rcpp::NumericMatrix const& x, int const k,
                           int const max_iters, double const tol,
                           int const init_method,
                           Rcpp::NumericMatrix const& centroids, int const seed,
                           int const verbosity) {
  Rcpp::List result;

  auto const m = Matrix<>(x, /*transpose=*/false);
  auto const n_samples = m.numRows;
  auto const n_features = m.numCols;

  ML::kmeans::KMeansParams params;
  params.n_clusters = k;
  // cuVS rejects tol=0, so continue one iteration at a time instead.
  bool const convergence_disabled = tol == 0;
  params.max_iter = convergence_disabled ? 1 : max_iters;
  params.tol = convergence_disabled ? 1e-4 : tol;
  params.init = static_cast<ML::kmeans::KMeansParams::InitMethod>(init_method);
  params.rng_state = raft::random::RngState(
    seed, raft::random::GeneratorType::GenPhilox);
  params.verbosity = static_cast<rapids_logger::level_enum>(verbosity);

  auto stream_view = stream_allocator::getOrCreateStream();
  raft::handle_t handle;
  handle_utils::initializeHandle(handle, stream_view.value());

  // kmeans input data
  auto const& h_src_data = m.values;

  auto const n_centroid_values = params.n_clusters * n_features;
  thrust::device_vector<double> d_src_data(h_src_data.size());
  auto CUML4R_ANONYMOUS_VARIABLE(src_data_h2d) =
    async_copy(stream_view.value(), h_src_data.cbegin(), h_src_data.cend(),
               d_src_data.begin());

  // kmeans outputs
  thrust::device_vector<double> d_pred_centroids(n_centroid_values);
  CUML4R_MAYBE_UNUSED AsyncCopyCtx centroids_h2d;
  if (params.init == ML::kmeans::KMeansParams::InitMethod::Array) {
    auto const m_centroids = Matrix<>(centroids, /*transpose=*/false);
    auto const& h_centroids = m_centroids.values;
    centroids_h2d = async_copy(stream_view.value(), h_centroids.cbegin(),
                               h_centroids.cend(), d_pred_centroids.begin());
  }
  thrust::device_vector<int> d_pred_labels(n_samples);

  double inertia = 0;
  int n_iter = 0;
  if (convergence_disabled) {
    for (int iteration = 0; iteration < max_iters; ++iteration) {
      ML::kmeans::fit(handle, params, d_src_data.data().get(), n_samples,
                      n_features, /*sample_weight=*/nullptr,
                      d_pred_centroids.data().get(), inertia, n_iter);
      params.init = ML::kmeans::KMeansParams::InitMethod::Array;
    }
    n_iter = max_iters;
  } else {
    ML::kmeans::fit(handle, params, d_src_data.data().get(), n_samples,
                    n_features, /*sample_weight=*/nullptr,
                    d_pred_centroids.data().get(), inertia, n_iter);
  }
  ML::kmeans::predict(handle, params, d_pred_centroids.data().get(),
                      d_src_data.data().get(), n_samples, n_features,
                      /*sample_weight=*/nullptr, /*normalize_weights=*/false,
                      d_pred_labels.data().get(), inertia);

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  pinned_host_vector<int> h_pred_labels(n_samples);
  auto CUML4R_ANONYMOUS_VARIABLE(pred_labels_d2h) =
    async_copy(stream_view.value(), d_pred_labels.cbegin(),
               d_pred_labels.cend(), h_pred_labels.begin());

  pinned_host_vector<double> h_pred_centroids(n_centroid_values);
  auto CUML4R_ANONYMOUS_VARIABLE(pred_centroids_d2h) =
    async_copy(stream_view.value(), d_pred_centroids.cbegin(),
               d_pred_centroids.cend(), h_pred_centroids.begin());

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  result["labels"] =
    Rcpp::IntegerVector(h_pred_labels.cbegin(), h_pred_labels.cend());
  result["centroids"] = Rcpp::transpose(
    Rcpp::NumericMatrix(n_features, k, h_pred_centroids.begin()));
  result["inertia"] = inertia;
  result["n_iter"] = n_iter;

  return result;
}

}  // namespace cuml4r
