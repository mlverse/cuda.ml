#if HAS_CUML

#include "cuda_utils.h"
#include "handle_utils.h"
#include "matrix_utils.h"
#include "stream_allocator.h"

#include <cuml/cluster/kmeans.hpp>

#include <memory>
#include <vector>

#endif

#include <Rcpp.h>


// [[Rcpp::export(".kmeans")]]
Rcpp::List kmeans(Rcpp::NumericMatrix const& m, int const k, int const max_iters) {
  Rcpp::List result;

#if HAS_CUML
  auto const matrix = cuml4r::Matrix<>(m, /*transpose=*/ true);
  auto const n_samples = matrix.numRows;
  auto const n_features = matrix.numCols;

  ML::kmeans::KMeansParams params;
  params.n_clusters = k;
  params.max_iter = max_iters;

  auto stream_view = cuml4r::stream_allocator::getOrCreateStream();
  auto stream = stream_view.value();
  raft::handle_t handle;
  cuml4r::handle_utils::initializeHandle(handle, stream);

  auto const& h_src_data = matrix.values;

  // kmeans input data
  double *d_src_data = nullptr;
  auto const src_data_sz = n_samples * n_features * sizeof(double);
  CUDA_RT_CALL(cudaMalloc(&d_src_data, src_data_sz));
  CUDA_RT_CALL(cudaMemcpyAsync(d_src_data, h_src_data.data(),
                               src_data_sz,
                               cudaMemcpyHostToDevice, stream));
  // kmeans outputs
  double *d_pred_centroids = nullptr;
  CUDA_RT_CALL(cudaMalloc(&d_pred_centroids,
                          params.n_clusters * n_features * sizeof(double)));
  int *d_pred_labels = nullptr;
  CUDA_RT_CALL(cudaMalloc(&d_pred_labels, n_samples * sizeof(int)));

  double inertia = 0;
  int n_iter = 0;
  ML::kmeans::fit_predict(handle, params, d_src_data, n_samples, n_features, 0,
                          d_pred_centroids, d_pred_labels, inertia, n_iter);

  std::vector<int> h_pred_labels(n_samples);
  CUDA_RT_CALL(cudaMemcpyAsync(h_pred_labels.data(), d_pred_labels,
                               n_samples * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
  std::vector<double> h_pred_centroids(params.n_clusters * n_features);
  CUDA_RT_CALL(
    cudaMemcpyAsync(h_pred_centroids.data(), d_pred_centroids,
                    params.n_clusters * n_features * sizeof(double),
                    cudaMemcpyDeviceToHost, stream));

  CUDA_RT_CALL(cudaStreamSynchronize(stream));

  result["labels"] = Rcpp::IntegerVector(h_pred_labels.cbegin(), h_pred_labels.cend());
  result["centroids"] = Rcpp::NumericMatrix(n_features, k, h_pred_centroids.begin());
  result["inertia"] = inertia;
  result["n_iter"] = n_iter;
#else

#include "warn_cuml_missing.h"

#endif

  return result;
}
