#include "async_utils.cuh"
#include "cuda_utils.h"
#include "fil_utils.h"
#include "handle_utils.h"
#include "matrix_utils.h"
#include "pinned_host_vector.h"
#include "preprocessor.h"
#include "random_forest.cuh"
#include "random_forest_serde.cuh"
#include "stream_allocator.h"

#include <thrust/device_vector.h>
#include <cuml/version_config.hpp>

#include <Rcpp.h>

#include <memory>
#include <unordered_map>
#include <vector>

namespace cuml4r {

namespace {

CUML4R_MAYBE_UNUSED constexpr auto kRfRegressorNumFeatures = "n_features";
CUML4R_MAYBE_UNUSED constexpr auto kRfRegressorForest = "forest";

using RandomForestRegressorUPtr =
  std::unique_ptr<ML::RandomForestRegressorD,
                  RandomForestMetaDataDeleter<double, double>>;

class RandomForestRegressor {
 public:
  __host__ RandomForestRegressor(RandomForestRegressorUPtr rf,
                                 int const n_features)
    : rf_(std::move(rf)), nFeatures_(n_features) {}

#ifndef CUML4R_TREELITE_C_API_MISSING
  __host__ explicit RandomForestRegressor(Rcpp::List const& state)
    : nFeatures_(state[kRfRegressorNumFeatures]), rf_(nullptr) {
    std::unique_ptr<treelite::Model> model;
    std::vector<detail::PyBufFrameContent> py_buf_frames_content;
    detail::setState(model, py_buf_frames_content,
                     /*state=*/state[kRfRegressorForest]);
    tlHandle_ = model.release();
    pyBufFramesContent_ = std::move(py_buf_frames_content);
  }

  __host__ Rcpp::List getState() {
    Rcpp::List state;

    state[kRfRegressorNumFeatures] = nFeatures_;

    auto const& treelite_handle = getTreeliteHandle();
    state[kRfRegressorForest] = detail::getState(
      *static_cast<treelite::Model const*>(treelite_handle.handle()));

    return state;
  }
#endif

  RandomForestRegressorUPtr const rf_;
  int const nFeatures_;

#ifndef CUML4R_TREELITE_C_API_MISSING

 public:
  TreeliteHandle const& getTreeliteHandle() {
    if (tlHandle_.empty()) {
      tlHandle_ = detail::build_treelite_forest(
        /*forest=*/rf_.get(),
        /*n_features=*/nFeatures_, /*n_classes=*/1);
    }

    return tlHandle_;
  }

 private:
  // `pyBufFramesContent_` is needed for storing non-POD PyBufFrame states on
  // the heap when calling setState(). It is otherwise unused.
  // NOTE: destruction of `tlHandle_` must precede the destruction of
  // `pyBufFrameContent_`.
  std::vector<detail::PyBufFrameContent> pyBufFramesContent_;
  TreeliteHandle tlHandle_;

#endif
};

template <typename InputT, typename OutputT>
__host__ Rcpp::NumericVector rf_regressor_predict(
  Rcpp::NumericMatrix const& input,
  std::function<void(raft::handle_t const&, InputT* const,
                     OutputT* const)> const& predict_impl) {
  auto const input_m = Matrix<InputT>(input, /*transpose=*/false);
  int const n_samples = input_m.numRows;

  auto stream_view = stream_allocator::getOrCreateStream();
  raft::handle_t handle;
  handle_utils::initializeHandle(handle, stream_view.value());

  // inputs
  auto const& h_input = input_m.values;
  thrust::device_vector<InputT> d_input(h_input.size());
  AsyncCopyCtx __attribute__((unused)) input_h2d;
  input_h2d = async_copy(stream_view.value(), h_input.cbegin(), h_input.cend(),
                         d_input.begin());

  // outputs
  thrust::device_vector<OutputT> d_preds(n_samples);

  predict_impl(handle, d_input.data().get(), d_preds.data().get());

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  pinned_host_vector<OutputT> h_preds(n_samples);
  AsyncCopyCtx __attribute__((unused)) preds_d2h;
  preds_d2h = async_copy(stream_view.value(), d_preds.cbegin(), d_preds.cend(),
                         h_preds.begin());

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  return Rcpp::NumericVector(h_preds.begin(), h_preds.end());
}

}  // namespace

__host__ SEXP rf_regressor_fit(
  Rcpp::NumericMatrix const& input, Rcpp::NumericVector const& responses,
  int const n_trees, bool const bootstrap, float const max_samples,
  int const n_streams, int const max_depth, int const max_leaves,
  float const max_features, int const n_bins, int const min_samples_leaf,
  int const min_samples_split, int const split_criterion,
  float const min_impurity_decrease, int const max_batch_size,
  int const verbosity) {
  auto const input_m = Matrix<>(input, /*transpose=*/true);
  int const n_samples = input_m.numCols;
  int const n_features = input_m.numRows;

  auto rf = RandomForestRegressorUPtr(new ML::RandomForestRegressorD);

  auto stream_view = stream_allocator::getOrCreateStream();
  raft::handle_t handle;
  handle_utils::initializeHandle(handle, stream_view.value());

  // rf input data & responses
  auto const& h_input = input_m.values;
  thrust::device_vector<double> d_input(h_input.size());
  auto CUML4R_ANONYMOUS_VARIABLE(input_h2d) = async_copy(
    stream_view.value(), h_input.cbegin(), h_input.cend(), d_input.begin());
  auto const h_responses(Rcpp::as<pinned_host_vector<double>>(responses));

  thrust::device_vector<double> d_responses(h_responses.size());
  auto CUML4R_ANONYMOUS_VARIABLE(responses_h2d) =
    async_copy(stream_view.value(), h_responses.cbegin(), h_responses.cend(),
               d_responses.begin());
  {
    auto* rf_ptr = rf.get();
    ML::fit(
      handle, rf_ptr, d_input.data().get(), n_samples, n_features,
      /*labels=*/d_responses.data().get(),
#if (CUML4R_LIBCUML_VERSION(CUML_VERSION_MAJOR, CUML_VERSION_MINOR) >= \
     CUML4R_LIBCUML_VERSION(21, 8))

      ML::set_rf_params(max_depth, max_leaves, max_features, n_bins,
                        min_samples_leaf, min_samples_split,
                        min_impurity_decrease, bootstrap, n_trees, max_samples,
                        /*seed=*/0, static_cast<ML::CRITERION>(split_criterion),
                        n_streams, max_batch_size),

#else

      ML::set_rf_params(
        max_depth, max_leaves, max_features, n_bins,
        /*split_algo=*/ML::SPLIT_ALGO::HIST, min_samples_leaf,
        min_samples_split, min_impurity_decrease,
        /*bootstrap_features=*/bootstrap, bootstrap, n_trees, max_samples,
        /*seed=*/0,
        /*split_criterion=*/static_cast<ML::CRITERION>(split_criterion),
        /*quantile_per_tree=*/false,
        /*cfg_n_streams=*/n_streams,
        /*use_experimental_backend=*/false, max_batch_size),

#endif
      /*verbosity=*/
#if (CUML4R_LIBCUML_VERSION(CUML_VERSION_MAJOR, CUML_VERSION_MINOR) >= \
     CUML4R_LIBCUML_VERSION(24, 0))
      static_cast<rapids_logger::level_enum>(verbosity)
#else
      verbosity
#endif
    );

    CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));
    if (rf_ptr != rf.get()) {
      // NOTE: in theory this should never happen though
      rf = RandomForestRegressorUPtr(rf_ptr);
    }
  }
  return Rcpp::XPtr<RandomForestRegressor>(
    std::make_unique<RandomForestRegressor>(/*rf=*/std::move(rf), n_features)
      .release());
}

__host__ Rcpp::NumericVector rf_regressor_predict(
  SEXP model_xptr, Rcpp::NumericMatrix const& input, int const verbosity) {
  int const n_samples = input.nrow();
  int const n_features = input.ncol();
  auto const model = Rcpp::XPtr<RandomForestRegressor>(model_xptr);

  if (model->rf_ != nullptr) {
    return rf_regressor_predict<double, double>(
      input,
      /*predict_impl=*/
      [n_samples, n_features, verbosity, rf = model->rf_.get()](
        raft::handle_t const& handle, double* const d_input,
        double* const d_preds) {
        ML::predict(handle, /*forest=*/rf, d_input, n_samples, n_features,
                    /*predictions=*/d_preds,
#if (CUML4R_LIBCUML_VERSION(CUML_VERSION_MAJOR, CUML_VERSION_MINOR) >= \
     CUML4R_LIBCUML_VERSION(24, 0))
                    static_cast<rapids_logger::level_enum>(verbosity)
#else
                    verbosity
#endif
        );
      });
  }

#ifndef CUML4R_TREELITE_C_API_MISSING
  auto const input_m = Matrix<float>(input, /*transpose=*/false);
  auto stream_view = stream_allocator::getOrCreateStream();
  raft::handle_t handle;
  handle_utils::initializeHandle(handle, stream_view.value());

  auto const& h_input = input_m.values;
  thrust::device_vector<float> d_input(h_input.size());
  auto CUML4R_ANONYMOUS_VARIABLE(input_h2d) = async_copy(
    stream_view.value(), h_input.cbegin(), h_input.cend(), d_input.begin());

  auto forest = fil::import_from_treelite(handle, model->getTreeliteHandle());
  auto const n_outputs = static_cast<size_t>(forest->num_outputs());
  thrust::device_vector<float> d_raw_predictions(n_samples * n_outputs);
  fil::predict(handle, *forest, d_raw_predictions.data().get(),
               d_input.data().get(), n_samples);

  pinned_host_vector<float> h_raw_predictions(d_raw_predictions.size());
  auto CUML4R_ANONYMOUS_VARIABLE(preds_d2h) =
    async_copy(stream_view.value(), d_raw_predictions.cbegin(),
               d_raw_predictions.cend(), h_raw_predictions.begin());

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  pinned_host_vector<float> h_predictions(n_samples);
  for (int i = 0; i < n_samples; ++i) {
    h_predictions[i] = h_raw_predictions[i * n_outputs];
  }

  return Rcpp::NumericVector(h_predictions.begin(), h_predictions.end());
#else
  Rcpp::stop(
    "Treelite C API is required for predictions using unserialized "
    "rand_forest model!");

#endif
}

__host__ Rcpp::List rf_regressor_get_state(SEXP model) {
#ifndef CUML4R_TREELITE_C_API_MISSING

  return Rcpp::XPtr<RandomForestRegressor>(model)->getState();

#else

  return {};

#endif
}

__host__ SEXP rf_regressor_set_state(Rcpp::List const& state) {
#ifndef CUML4R_TREELITE_C_API_MISSING

  auto model = std::make_unique<RandomForestRegressor>(state);
  return Rcpp::XPtr<RandomForestRegressor>(model.release());

#else

  return R_NilValue;

#endif
}

}  // namespace cuml4r
