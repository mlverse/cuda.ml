#include "async_utils.cuh"
#include "cuda_utils.h"
#include "handle_utils.h"
#include "matrix_utils.h"
#include "pinned_host_vector.h"
#include "preprocessor.h"
#include "stream_allocator.h"

#include <thrust/async/copy.h>
#include <thrust/device_vector.h>
#include <cuml/cluster/linkage.hpp>

#include <Rcpp.h>

#include <memory>

namespace cuml4r {

__host__ Rcpp::List agglomerative_clustering(Rcpp::NumericMatrix const& x,
                                             bool const pairwise_conn,
                                             int const metric,
                                             int const n_neighbors,
                                             int const n_clusters) {
  Rcpp::List result;

  auto const m = Matrix<float>(x, /*transpose=*/false);
  auto const n_samples = m.numRows;
  auto const n_features = m.numCols;

  auto stream_view = stream_allocator::getOrCreateStream();
  raft::handle_t handle;
  handle_utils::initializeHandle(handle, stream_view.value());

  // single-linkage hierarchical clustering input
  auto const& h_x = m.values;
  thrust::device_vector<float> d_x(h_x.size());
  auto CUML4R_ANONYMOUS_VARIABLE(x_h2d) =
    async_copy(stream_view.value(), h_x.cbegin(), h_x.cend(), d_x.begin());

  // single-linkage hierarchical clustering output
  auto out = std::make_unique<raft::hierarchy::linkage_output<int, float>>();
  thrust::device_vector<int> d_labels(n_samples);
  thrust::device_vector<int> d_children((n_samples - 1) * 2);
  out->labels = d_labels.data().get();
  out->children = d_children.data().get();

  if (pairwise_conn) {
    ML::single_linkage_pairwise(
      handle, /*X=*/d_x.data().get(), /*m=*/n_samples, /*n=*/n_features,
      /*out=*/out.get(),
      /*metric=*/static_cast<raft::distance::DistanceType>(metric), n_clusters);
  } else {
    ML::single_linkage_neighbors(
      handle, /*X=*/d_x.data().get(), /*m=*/n_samples, /*n=*/n_features,
      /*out=*/out.get(),
      /*metric=*/static_cast<raft::distance::DistanceType>(metric),
      /*c=*/n_neighbors, n_clusters);
  }

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  pinned_host_vector<int> h_labels(d_labels.size());
  pinned_host_vector<int> h_children(d_children.size());
  auto CUML4R_ANONYMOUS_VARIABLE(labels_d2h) = async_copy(
    stream_view.value(), d_labels.cbegin(), d_labels.cend(), h_labels.begin());
  auto CUML4R_ANONYMOUS_VARIABLE(children_d2h) =
    async_copy(stream_view.value(), d_children.cbegin(), d_children.cend(),
               h_children.begin());

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  result["n_clusters"] = out->n_clusters;
  result["children"] =
    Rcpp::transpose(Rcpp::IntegerMatrix(2, n_samples - 1, h_children.begin()));
  result["labels"] = Rcpp::IntegerVector(h_labels.cbegin(), h_labels.cend());

  return result;
}

}  // namespace cuml4r
