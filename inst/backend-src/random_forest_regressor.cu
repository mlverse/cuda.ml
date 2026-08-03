#include "random_forest_regressor.h"

#include "async_utils.cuh"
#include "cuda_utils.h"
#include "handle_utils.h"
#include "matrix_utils.h"
#include "nvforest.h"
#include "nvforest_internal.h"
#include "preprocessor.h"
#include "stream_allocator.h"
#include "treelite_utils.cuh"

#include <thrust/device_vector.h>
#include <cuml/ensemble/randomforest.hpp>
#include <rapids_logger/logger.hpp>

#include <Rcpp.h>

#include <cstdint>
#include <vector>

namespace cuml4r {
namespace {

ML::RF_params rf_params(int const n_trees, bool const bootstrap,
                        float const max_samples, int const n_streams,
                        int const max_depth, int const max_leaves,
                        float const max_features, int const n_bins,
                        int const min_samples_leaf, int const min_samples_split,
                        int const split_criterion,
                        float const min_impurity_decrease,
                        int const max_batch_size, int const seed) {
  if (seed < 0) {
    Rcpp::stop("'seed' must be a non-negative integer.");
  }
  return ML::set_rf_params(
    max_depth, max_leaves, max_features, n_bins, min_samples_leaf,
    min_samples_split, min_impurity_decrease, bootstrap, n_trees, max_samples,
    static_cast<std::uint64_t>(seed),
    static_cast<ML::CRITERION>(split_criterion), n_streams, max_batch_size);
}

}  // namespace

SEXP rf_regressor_fit(Rcpp::NumericMatrix const& input,
                      Rcpp::NumericVector const& responses, int const n_trees,
                      bool const bootstrap, float const max_samples,
                      int const n_streams, int const max_depth,
                      int const max_leaves, float const max_features,
                      int const n_bins, int const min_samples_leaf,
                      int const min_samples_split, int const split_criterion,
                      float const min_impurity_decrease,
                      int const max_batch_size, int const seed) {
  auto const input_matrix = Matrix<>(input, /*transpose=*/true);
  auto const n_samples = static_cast<int>(input_matrix.numCols);
  auto const n_features = static_cast<int>(input_matrix.numRows);
  if (responses.size() != n_samples) {
    Rcpp::stop("'responses' must have one value for each input row.");
  }
  auto const params =
    rf_params(n_trees, bootstrap, max_samples, n_streams, max_depth, max_leaves,
              max_features, n_bins, min_samples_leaf, min_samples_split,
              split_criterion, min_impurity_decrease, max_batch_size, seed);

  auto const stream = stream_allocator::getOrCreateStream();
  raft::handle_t handle;
  handle_utils::initializeHandle(handle, stream, n_streams);

  thrust::device_vector<double> device_input(input_matrix.values.size());
  auto CUML4R_ANONYMOUS_VARIABLE(input_copy) =
    async_copy(stream.value(), input_matrix.values.cbegin(),
               input_matrix.values.cend(), device_input.begin());
  auto const host_responses = Rcpp::as<std::vector<double>>(responses);
  thrust::device_vector<double> device_responses(host_responses.size());
  auto CUML4R_ANONYMOUS_VARIABLE(response_copy) =
    async_copy(stream.value(), host_responses.cbegin(), host_responses.cend(),
               device_responses.begin());

  TreeliteHandle treelite;
  ML::fit_treelite<double, double>(
    handle, treelite.out(), device_input.data().get(), n_samples, n_features,
    device_responses.data().get(), params, /*bootstrap_masks=*/nullptr,
    /*feature_importances=*/nullptr, rapids_logger::level_enum::off);
  handle.sync_stream();

  return nvforest_from_treelite(
    std::move(treelite),
    NvForestOptions{NvForestDevice::GPU, -1, NvForestLayout::DEPTH_FIRST,
                    NvForestPrecision::NATIVE, 0, -1},
    /*averaged_vector_leaf_probabilities=*/false);
}

}  // namespace cuml4r
