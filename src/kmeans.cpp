#if HAS_CUML

#include "async_utils.h"
#include "cuda_utils.h"
#include "handle_utils.h"
#include "matrix_utils.h"
#include "pinned_host_vector.h"
#include "preprocessor.h"
#include "stream_allocator.h"

#include <thrust/async/copy.h>
#include <thrust/device_vector.h>
#include <cuml/cluster/kmeans.hpp>

#include <memory>
#include <vector>

#else

#include "warn_cuml_missing.h"

#endif

#include <Rcpp.h>

// [[Rcpp::export(".kmeans")]]
Rcpp::List kmeans(Rcpp::NumericMatrix const& x, int const k,
                  int const max_iters) {
  Rcpp::List result;

#if HAS_CUML
  auto const m = cuml4r::Matrix<>(x, /*transpose=*/false);
  auto const n_samples = m.numRows;
  auto const n_features = m.numCols;

  ML::kmeans::KMeansParams params;
  params.n_clusters = k;
  params.max_iter = max_iters;

  auto stream_view = cuml4r::stream_allocator::getOrCreateStream();
  raft::handle_t handle;
  cuml4r::handle_utils::initializeHandle(handle, stream_view.value());

  // kmeans input data
  auto const& h_src_data = m.values;

  auto const n_centroid_values = params.n_clusters * n_features;
  thrust::device_vector<double> d_src_data(h_src_data.size());
  auto CUML4R_ANONYMOUS_VARIABLE(src_data_h2d) =
    cuml4r::async_copy(stream_view.value(), h_src_data.cbegin(),
                       h_src_data.cend(), d_src_data.begin());

  // kmeans outputs
  thrust::device_vector<double> d_pred_centroids(n_centroid_values);
  thrust::device_vector<int> d_pred_labels(n_samples);

  double inertia = 0;
  int n_iter = 0;
  ML::kmeans::fit_predict(handle, params, d_src_data.data().get(), n_samples,
                          n_features, 0, d_pred_centroids.data().get(),
                          d_pred_labels.data().get(), inertia, n_iter);

  cuml4r::pinned_host_vector<int> h_pred_labels(n_samples);
  auto CUML4R_ANONYMOUS_VARIABLE(pred_labels_d2h) =
    cuml4r::async_copy(stream_view.value(), d_pred_labels.cbegin(),
                       d_pred_labels.cend(), h_pred_labels.begin());

  cuml4r::pinned_host_vector<double> h_pred_centroids(n_centroid_values);
  auto CUML4R_ANONYMOUS_VARIABLE(pred_centroids_d2h) =
    cuml4r::async_copy(stream_view.value(), d_pred_centroids.cbegin(),
                       d_pred_centroids.cend(), h_pred_centroids.begin());

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  result["labels"] =
    Rcpp::IntegerVector(h_pred_labels.cbegin(), h_pred_labels.cend());
  result["centroids"] = Rcpp::transpose(
    Rcpp::NumericMatrix(n_features, k, h_pred_centroids.begin()));
  result["inertia"] = inertia;
  result["n_iter"] = n_iter;
#else

#include "warn_cuml_missing.h"

#endif

  return result;
}
