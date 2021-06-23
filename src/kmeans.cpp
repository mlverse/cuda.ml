#include <cstring>

#if HAS_CUML
#include <raft/handle.hpp>
#include <raft/mr/device/allocator.hpp>
#include <cuml/cluster/kmeans.hpp>
#include "cuda_utils.h"
#else
#include <Rcpp.h>
#endif

// [[Rcpp::export]]
Rcpp::List kMeans(Rcpp::NumericMatrix const& m, int const k, int const max_iters) {
  Rcpp::List result;

#if HAS_CUML
  auto const n_samples = m.ncol();
  auto const n_features = m.nrow();
  
  ML::kmeans::KMeansParams params;
  params.n_clusters = k;
  params.max_iter = max_iters;
  
  raft::handle_t handle;
  auto const allocator = std::make_shared<raft::mr::device::default_allocator>();
  handle.set_device_allocator(allocator);
  
  cudaStream_t stream;
  CUDA_RT_CALL(cudaStreamCreate(&stream));
  handle.set_stream(stream);
  
  auto const h_src_data = Rcpp::as<std::vector<double>>(Rcpp::NumericVector(m));
  
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
#endif

  return result;
}
