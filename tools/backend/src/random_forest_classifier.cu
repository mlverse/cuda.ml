#include "random_forest_classifier.h"

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

#include <algorithm>
#include <cstdint>
#include <vector>

namespace cuml4r {
namespace {

int validate_labels(Rcpp::IntegerVector const& labels, int const n_samples) {
  if (labels.size() != n_samples) {
    Rcpp::stop("'labels' must have one value for each input row.");
  }
  if (labels.size() == 0) {
    Rcpp::stop("Random forest classification requires at least one row.");
  }

  auto const max_label = *std::max_element(labels.cbegin(), labels.cend());
  if (max_label < 1) {
    Rcpp::stop("Random forest classification requires at least two classes.");
  }
  auto seen = std::vector<bool>(static_cast<std::size_t>(max_label) + 1);
  for (auto const label : labels) {
    if (label == NA_INTEGER || label < 0) {
      Rcpp::stop("'labels' must be contiguous zero-based integers.");
    }
    seen[static_cast<std::size_t>(label)] = true;
  }
  if (std::find(seen.cbegin(), seen.cend(), false) != seen.cend()) {
    Rcpp::stop("'labels' must be contiguous zero-based integers.");
  }
  return max_label + 1;
}

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

SEXP rf_classifier_fit(Rcpp::NumericMatrix const& input,
                       Rcpp::IntegerVector const& labels, int const n_trees,
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
  auto const n_classes = validate_labels(labels, n_samples);
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
  auto const host_labels = Rcpp::as<std::vector<int>>(labels);
  thrust::device_vector<int> device_labels(host_labels.size());
  auto CUML4R_ANONYMOUS_VARIABLE(label_copy) =
    async_copy(stream.value(), host_labels.cbegin(), host_labels.cend(),
               device_labels.begin());

  TreeliteHandle treelite;
  ML::fit_treelite<double, int>(
    handle, treelite.out(), device_input.data().get(), n_samples, n_features,
    device_labels.data().get(), n_classes, params,
    /*bootstrap_masks=*/nullptr, /*feature_importances=*/nullptr,
    rapids_logger::level_enum::off);
  handle.sync_stream();

  return nvforest_from_treelite(
    std::move(treelite),
    NvForestOptions{NvForestDevice::GPU, -1, NvForestLayout::DEPTH_FIRST,
                    NvForestPrecision::NATIVE, 0, -1},
    /*averaged_vector_leaf_probabilities=*/true);
}

}  // namespace cuml4r
