#if HAS_CUML

#include "random_forest.h"
#include "async_utils.h"
#include "cuda_utils.h"
#include "handle_utils.h"
#include "matrix_utils.h"
#include "pinned_host_vector.h"
#include "preprocessor.h"
#include "stream_allocator.h"

#include <thrust/async/copy.h>
#include <thrust/device_vector.h>
#include <cuml/ensemble/randomforest.hpp>

#include <chrono>
#include <memory>
#include <unordered_map>
#include <vector>

#endif

#include <Rcpp.h>

#if HAS_CUML

namespace {

// map labels into consecutive integral values in [0, to n_unique_labels)
__host__ cuml4r::pinned_host_vector<int> preprocess_labels(
  std::unordered_map<int, int>& labels_map, std::vector<int> const& labels) {
  int n_unique_labels = 0;
  cuml4r::pinned_host_vector<int> preprocessed_labels;
  preprocessed_labels.reserve(labels.size());

  for (auto const label : labels) {
    auto const p = labels_map.emplace(label, n_unique_labels);
    if (p.second) {
      ++n_unique_labels;
    }
    preprocessed_labels.push_back(p.first->second);
  }

  return preprocessed_labels;
}

// reverse the mapping done by preprocess_labels
__host__ void postprocess_labels(
  cuml4r::pinned_host_vector<int>& labels,
  std::unordered_map<int, int> const& inverse_labels_map) {
  for (auto& label : labels) {
    auto iter = inverse_labels_map.find(label);
    if (iter != inverse_labels_map.cend()) {
      label = iter->second;
    } else {
      label = 0;
    }
  }
}

__host__ std::unordered_map<int, int> reverse(
  std::unordered_map<int, int> const& m) {
  std::unordered_map<int, int> r;
  r.reserve(m.size());
  for (auto const& p : m) {
    r[p.second] = p.first;
  }
  return r;
}

}  // namespace

#else

#include "warn_cuml_missing.h"

#endif

// fit and predict functions for classifiers

// [[Rcpp::export(".rf_classifier_fit")]]
SEXP rf_classifier_fit(Rcpp::NumericMatrix const& input,
                       Rcpp::IntegerVector const& labels, int const n_trees,
                       bool const bootstrap, float const max_samples,
                       int const n_streams, int const max_depth,
                       int const max_leaves, float const max_features,
                       int const n_bins, int const min_samples_leaf,
                       int const min_samples_split, int const split_criterion,
                       float const min_impurity_decrease,
                       int const max_batch_size, int const verbosity) {
#if HAS_CUML
  auto const input_m = cuml4r::Matrix<>(input, /*transpose=*/true);
  int const n_samples = input_m.numCols;
  int const n_features = input_m.numRows;

  auto rf = cuml4r::RandomForestClassifierUPtr(new ML::RandomForestClassifierD);

  auto stream_view = cuml4r::stream_allocator::getOrCreateStream();
  raft::handle_t handle(n_streams);
  cuml4r::handle_utils::initializeHandle(handle, stream_view.value());

  // rf input data & labels
  auto const& h_input = input_m.values;
  thrust::device_vector<double> d_input(h_input.size());
  auto CUML4R_ANONYMOUS_VARIABLE(input_h2d) = cuml4r::async_copy(
    stream_view.value(), h_input.cbegin(), h_input.cend(), d_input.begin());
  std::unordered_map<int, int> labels_map;
  auto const h_labels =
    preprocess_labels(labels_map, Rcpp::as<std::vector<int>>(labels));

  thrust::device_vector<int> d_labels(h_labels.size());
  auto CUML4R_ANONYMOUS_VARIABLE(labels_h2d) = cuml4r::async_copy(
    stream_view.value(), h_labels.cbegin(), h_labels.cend(), d_labels.begin());
  {
    auto* rf_ptr = rf.get();
    ML::fit(
      handle, rf_ptr, d_input.data().get(), n_samples, n_features,
      d_labels.data().get(),
      /*n_unique_labels=*/static_cast<int>(labels_map.size()),
      ML::set_rf_params(max_depth, max_leaves, max_features, n_bins,
                        min_samples_leaf, min_samples_split,
                        min_impurity_decrease, bootstrap, n_trees, max_samples,
                        std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count(),
                        static_cast<ML::CRITERION>(split_criterion), n_streams,
                        max_batch_size),
      /*verbosity=*/verbosity);

    CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));
    if (rf_ptr != rf.get()) {
      // NOTE: in theory this should never happen though
      rf = cuml4r::RandomForestClassifierUPtr(rf_ptr);
    }
  }

  return Rcpp::XPtr<cuml4r::RandomForestClassifierModel>(
    new cuml4r::RandomForestClassifierModel(std::move(rf),
                                            reverse(labels_map)));
#else

#include "warn_cuml_missing.h"

  return Rcpp::List();

#endif
}

// [[Rcpp::export(".rf_classifier_predict")]]
Rcpp::IntegerVector rf_classifier_predict(SEXP model_xptr,
                                          Rcpp::NumericMatrix const& input,
                                          int const verbosity) {
#if HAS_CUML
  auto const input_m = cuml4r::Matrix<>(input, /*transpose=*/false);
  int const n_samples = input_m.numRows;
  int const n_features = input_m.numCols;

  auto stream_view = cuml4r::stream_allocator::getOrCreateStream();
  raft::handle_t handle;
  cuml4r::handle_utils::initializeHandle(handle, stream_view.value());

  auto model = Rcpp::XPtr<cuml4r::RandomForestClassifierModel>(model_xptr);

  // inputs
  auto const& h_input = input_m.values;
  thrust::device_vector<double> d_input(h_input.size());
  auto CUML4R_ANONYMOUS_VARIABLE(input_h2d) = cuml4r::async_copy(
    stream_view.value(), h_input.cbegin(), h_input.cend(), d_input.begin());

  // outputs
  thrust::device_vector<int> d_predictions(n_samples);

  ML::predict(handle, /*forest=*/model->rf_.get(), d_input.data().get(),
              n_samples, n_features, /*predictions=*/d_predictions.data().get(),
              /*verbosity=*/verbosity);

  cuml4r::pinned_host_vector<int> h_predictions(n_samples);
  auto CUML4R_ANONYMOUS_VARIABLE(predictions_d2h) =
    cuml4r::async_copy(stream_view.value(), d_predictions.cbegin(),
                       d_predictions.cend(), h_predictions.begin());

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  // post-process prediction labels
  postprocess_labels(h_predictions, model->inverseLabelsMap_);

  return Rcpp::IntegerVector(h_predictions.begin(), h_predictions.end());
#else

#include "warn_cuml_missing.h"

  return {};

#endif
}

// fit and predict functions for regressors

// [[Rcpp::export(".rf_regressor_fit")]]
SEXP rf_regressor_fit(Rcpp::NumericMatrix const& input,
                      Rcpp::NumericVector const& responses, int const n_trees,
                      bool const bootstrap, float const max_samples,
                      int const n_streams, int const max_depth,
                      int const max_leaves, float const max_features,
                      int const n_bins, int const min_samples_leaf,
                      int const min_samples_split, int const split_criterion,
                      float const min_impurity_decrease,
                      int const max_batch_size, int const verbosity) {
#if HAS_CUML
  auto const input_m = cuml4r::Matrix<>(input, /*transpose=*/true);
  int const n_samples = input_m.numCols;
  int const n_features = input_m.numRows;

  auto rf = cuml4r::RandomForestRegressorUPtr(new ML::RandomForestRegressorD);

  auto stream_view = cuml4r::stream_allocator::getOrCreateStream();
  raft::handle_t handle(n_streams);
  cuml4r::handle_utils::initializeHandle(handle, stream_view.value());

  // rf input data & responses
  auto const& h_input = input_m.values;
  thrust::device_vector<double> d_input(h_input.size());
  auto CUML4R_ANONYMOUS_VARIABLE(input_h2d) = cuml4r::async_copy(
    stream_view.value(), h_input.cbegin(), h_input.cend(), d_input.begin());
  cuml4r::pinned_host_vector<double> h_responses(Rcpp::as<std::vector<double>>(responses));

  thrust::device_vector<double> d_responses(h_responses.size());
  auto CUML4R_ANONYMOUS_VARIABLE(responses_h2d) = cuml4r::async_copy(
    stream_view.value(), h_responses.cbegin(), h_responses.cend(), d_responses.begin());
  {
    auto* rf_ptr = rf.get();
    ML::fit(
      handle, rf_ptr, d_input.data().get(), n_samples, n_features,
      /*labels=*/d_responses.data().get(),
      ML::set_rf_params(max_depth, max_leaves, max_features, n_bins,
                        min_samples_leaf, min_samples_split,
                        min_impurity_decrease, bootstrap, n_trees, max_samples,
                        std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count(),
                        static_cast<ML::CRITERION>(split_criterion), n_streams,
                        max_batch_size),
      /*verbosity=*/verbosity);

    CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));
    if (rf_ptr != rf.get()) {
      // NOTE: in theory this should never happen though
      rf = cuml4r::RandomForestRegressorUPtr(rf_ptr);
    }
  }

  return Rcpp::XPtr<ML::RandomForestRegressorD>(rf.release());
#else

#include "warn_cuml_missing.h"

  return Rcpp::List();

#endif
}

// [[Rcpp::export(".rf_regressor_predict")]]
Rcpp::NumericVector rf_regressor_predict(SEXP model_xptr,
                                          Rcpp::NumericMatrix const& input,
                                          int const verbosity) {
#if HAS_CUML
  auto const input_m = cuml4r::Matrix<>(input, /*transpose=*/false);
  int const n_samples = input_m.numRows;
  int const n_features = input_m.numCols;

  auto stream_view = cuml4r::stream_allocator::getOrCreateStream();
  raft::handle_t handle;
  cuml4r::handle_utils::initializeHandle(handle, stream_view.value());

  auto model = Rcpp::XPtr<ML::RandomForestRegressorD>(model_xptr);

  // inputs
  auto const& h_input = input_m.values;
  thrust::device_vector<double> d_input(h_input.size());
  auto CUML4R_ANONYMOUS_VARIABLE(input_h2d) = cuml4r::async_copy(
    stream_view.value(), h_input.cbegin(), h_input.cend(), d_input.begin());

  // outputs
  thrust::device_vector<double> d_predictions(n_samples);

  ML::predict(handle, /*forest=*/model.get(), d_input.data().get(),
              n_samples, n_features, /*predictions=*/d_predictions.data().get(),
              /*verbosity=*/verbosity);

  cuml4r::pinned_host_vector<double> h_predictions(n_samples);
  auto CUML4R_ANONYMOUS_VARIABLE(predictions_d2h) =
    cuml4r::async_copy(stream_view.value(), d_predictions.cbegin(),
                       d_predictions.cend(), h_predictions.begin());

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  return Rcpp::NumericVector(h_predictions.begin(), h_predictions.end());
#else

#include "warn_cuml_missing.h"

  return {};

#endif
}
