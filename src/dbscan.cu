#include "async_utils.cuh"
#include "cuda_utils.h"
#include "handle_utils.h"
#include "matrix_utils.h"
#include "preprocessor.h"
#include "stream_allocator.h"

#include <thrust/async/copy.h>
#include <thrust/device_vector.h>
#include <cuml/cluster/dbscan.hpp>

#include <Rcpp.h>

#include <memory>

namespace cuml4r {

__host__ Rcpp::List dbscan(Rcpp::NumericMatrix const& x, int const min_pts,
                           double const eps, size_t const max_bytes_per_batch,
                           int const verbosity) {
  Rcpp::List result;

  auto const m = Matrix<>(x, /*transpose=*/false);
  auto const n_samples = m.numRows;
  auto const n_features = m.numCols;
  auto const& h_src_data = m.values;

  auto stream_view = stream_allocator::getOrCreateStream();
  raft::handle_t handle;
  handle_utils::initializeHandle(handle, stream_view.value());

  // dbscan input data
  thrust::device_vector<double> d_src_data(h_src_data.size());

  // dbscan output data
  thrust::device_vector<int> d_labels(n_samples);

  auto CUML4R_ANONYMOUS_VARIABLE(src_data_h2d) =
    async_copy(stream_view.value(), h_src_data.cbegin(), h_src_data.cend(),
               d_src_data.begin());

  ML::Dbscan::fit(handle, /*input=*/d_src_data.data().get(),
                  /*n_rows=*/n_samples, /*n_cols=*/n_features, eps, min_pts,
                  /*metric=*/raft::distance::L2SqrtUnexpanded,
                  /*labels=*/d_labels.data().get(),
                  /*core_sample_indices=*/nullptr, max_bytes_per_batch,
                  /*verbosity=*/verbosity, /*opg=*/false);

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  pinned_host_vector<int> h_labels(n_samples);
  auto CUML4R_ANONYMOUS_VARIABLE(labels_d2h) = async_copy(
    stream_view.value(), d_labels.cbegin(), d_labels.cend(), h_labels.begin());

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  result["labels"] = Rcpp::IntegerVector(h_labels.cbegin(), h_labels.cend());

  return result;
}

}  // namespace cuml4r
