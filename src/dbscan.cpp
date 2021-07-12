#if HAS_CUML

#include "cuda_utils.h"
#include "handle_utils.h"
#include "matrix_utils.h"
#include "stream_allocator.h"

#include <cuml/cluster/dbscan.hpp>

#include <memory>
#include <vector>

#endif

#include <Rcpp.h>


// [[Rcpp::export(".dbscan")]]
Rcpp::List dbscan(Rcpp::NumericMatrix const& m, int const min_pts, float const eps, size_t max_bytes_per_batch) {
  Rcpp::List result;

#if HAS_CUML
  auto const matrix = cuml4r::Matrix<float>(m, /*transpose=*/ true);
  auto const n_samples = matrix.numRows;
  auto const n_features = matrix.numCols;
  auto const& h_src_data = matrix.values;

  auto stream_view = cuml4r::stream_allocator::getOrCreateStream();
  auto stream = stream_view.value();
  raft::handle_t handle;
  cuml4r::handle_utils::initializeHandle(handle, stream);

  // dbscan input data
  float *d_src_data = nullptr;

  // dbscan output data
  int* d_labels = nullptr;

  auto const labels_sz = n_samples * sizeof(int);
  CUDA_RT_CALL(cudaMalloc(&d_labels, labels_sz));

  auto const src_data_sz = n_samples * n_features * sizeof(float);
  CUDA_RT_CALL(cudaMalloc(&d_src_data, src_data_sz));
  CUDA_RT_CALL(cudaMemcpyAsync(d_src_data, h_src_data.data(),
                               src_data_sz,
                               cudaMemcpyHostToDevice, stream));

  ML::Dbscan::fit(handle, d_src_data, n_samples, n_features, eps, min_pts,
                  raft::distance::L2SqrtUnexpanded, d_labels, nullptr,
                  max_bytes_per_batch, false);

  std::vector<int> h_labels(n_samples);
  CUDA_RT_CALL(cudaMemcpyAsync(h_labels.data(), d_labels, labels_sz,
                               cudaMemcpyDeviceToHost, stream));
  CUDA_RT_CALL(cudaStreamSynchronize(stream));

  result["labels"] = Rcpp::IntegerVector(h_labels.cbegin(), h_labels.cend());
#else

#include "warn_cuml_missing.h"

#endif

  return result;
}
