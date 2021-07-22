#if HAS_CUML

#include "async_utils.h"
#include "cuda_utils.h"
#include "handle_utils.h"
#include "matrix_utils.h"
#include "pinned_host_vector.h"
#include "preprocessor.h"
#include "random_forest.h"
#include "stream_allocator.h"

#include <thrust/async/copy.h>
#include <thrust/device_vector.h>
#include <cuml/ensemble/randomforest.hpp>

#include <chrono>
#include <memory>
#include <unordered_map>
#include <vector>

#else

#include "warn_cuml_missing.h"

#endif

#include <Rcpp.h>

#if HAS_CUML
using RandomForestRegressorUPtr =
  std::unique_ptr<ML::RandomForestRegressorD,
                  cuml4r::RandomForestMetaDataDeleter<double, double>>;

#else

#include "warn_cuml_missing.h"

#endif

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

  auto rf = RandomForestRegressorUPtr(new ML::RandomForestRegressorD);

  auto stream_view = cuml4r::stream_allocator::getOrCreateStream();
  raft::handle_t handle(n_streams);
  cuml4r::handle_utils::initializeHandle(handle, stream_view.value());

  // rf input data & responses
  auto const& h_input = input_m.values;
  thrust::device_vector<double> d_input(h_input.size());
  auto CUML4R_ANONYMOUS_VARIABLE(input_h2d) = cuml4r::async_copy(
    stream_view.value(), h_input.cbegin(), h_input.cend(), d_input.begin());
  cuml4r::pinned_host_vector<double> h_responses(
    Rcpp::as<std::vector<double>>(responses));

  thrust::device_vector<double> d_responses(h_responses.size());
  auto CUML4R_ANONYMOUS_VARIABLE(responses_h2d) =
    cuml4r::async_copy(stream_view.value(), h_responses.cbegin(),
                       h_responses.cend(), d_responses.begin());
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
      rf = RandomForestRegressorUPtr(rf_ptr);
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

  ML::predict(handle, /*forest=*/model.get(), d_input.data().get(), n_samples,
              n_features, /*predictions=*/d_predictions.data().get(),
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
