#if HAS_CUML

#include "async_utils.h"
#include "cuda_utils.h"
#include "handle_utils.h"
#include "matrix_utils.h"
#include "preprocessor.h"
#include "stream_allocator.h"

#include <thrust/async/copy.h>
#include <thrust/device_vector.h>
#include <cuml/cluster/dbscan.hpp>

#include <memory>
#include <vector>

#endif

#include <Rcpp.h>

// [[Rcpp::export(".dbscan")]]
Rcpp::List dbscan(Rcpp::NumericMatrix const& x, int const min_pts,
                  double const eps, size_t const max_bytes_per_batch) {
  Rcpp::List result;

#if HAS_CUML
  auto const m = cuml4r::Matrix<>(x, /*transpose=*/false);
  auto const n_samples = m.numRows;
  auto const n_features = m.numCols;
  auto const& h_src_data = m.values;

  auto stream_view = cuml4r::stream_allocator::getOrCreateStream();
  raft::handle_t handle;
  cuml4r::handle_utils::initializeHandle(handle, stream_view.value());

  // dbscan input data
  thrust::device_vector<double> d_src_data(h_src_data.size());

  // dbscan output data
  thrust::device_vector<int> d_labels(n_samples);

  auto CUML4R_ANONYMOUS_VARIABLE(src_data_h2d) =
    cuml4r::async_copy(stream_view.value(), h_src_data.cbegin(),
                       h_src_data.cend(), d_src_data.begin());

  ML::Dbscan::fit(handle, d_src_data.data().get(), n_samples, n_features, eps,
                  min_pts, raft::distance::L2SqrtUnexpanded,
                  d_labels.data().get(), nullptr, max_bytes_per_batch, false);

  std::vector<int> h_labels(n_samples);
  auto CUML4R_ANONYMOUS_VARIABLE(labels_d2h) = cuml4r::async_copy(
    stream_view.value(), d_labels.cbegin(), d_labels.cend(), h_labels.begin());

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  result["labels"] = Rcpp::IntegerVector(h_labels.cbegin(), h_labels.cend());
#else

#include "warn_cuml_missing.h"

#endif

  return result;
}
