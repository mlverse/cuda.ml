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

#include <thrust/async/copy.h>
#include <thrust/device_vector.h>
#include <cuml/version_config.hpp>

#include <Rcpp.h>

#include <memory>
#include <unordered_map>
#include <vector>

namespace cuml4r {

namespace {

constexpr auto kRfRegressorNumFeatures = "n_features";
constexpr auto kRfRegressorForest = "forest";

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
      *reinterpret_cast<treelite::Model const*>(*treelite_handle.get()));

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

/*
 * The 'ML::fil::treelite_params_t::threads_per_tree' and
 * 'ML::fil::treelite_params_t::n_items' parameters are only supported in
 * RAPIDS cuML 21.08 or above.
 */
CUML4R_ASSIGN_IF_PRESENT(threads_per_tree)
CUML4R_NOOP_IF_ABSENT(threads_per_tree)

CUML4R_ASSIGN_IF_PRESENT(n_items)
CUML4R_NOOP_IF_ABSENT(n_items)

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
  raft::handle_t handle(n_streams);
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
#if CUML4R_CONCAT(0x, CUML_VERSION_MINOR) >= 0x08

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
      /*verbosity=*/verbosity);

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
                    /*predictions=*/d_preds, verbosity);
      });
  } else {
    return rf_regressor_predict<float, float>(
      input,
      /*predict_impl=*/
      [&model, n_samples, n_features](raft::handle_t const& handle,
                                      float* const d_input,
                                      float* const d_preds) {
#ifndef CUML4R_TREELITE_C_API_MISSING
        auto const& tl_handle = model->getTreeliteHandle();

        ML::fil::treelite_params_t params;
        params.algo = ML::fil::algo_t::ALGO_AUTO;
        params.output_class = false;
        params.storage_type = ML::fil::storage_type_t::AUTO;
        params.blocks_per_sm = 0;
        set_threads_per_tree(params, 1);
        set_n_items(params, 0);
        params.pforest_shape_str = nullptr;
        auto forest =
          fil::make_forest(handle,
                           /*src=*/[&] {
                             ML::fil::forest* f;
                             ML::fil::from_treelite(handle, /*pforest=*/&f,
                                                    /*model=*/*tl_handle.get(),
                                                    /*tl_params=*/&params);
                             return f;
                           });
        ML::fil::predict(/*h=*/handle, /*f=*/forest.get(), /*preds=*/d_preds,
                         /*data=*/d_input, /*num_rows=*/n_samples,
                         /*predict_proba=*/false);

#else
        Rcpp::stop(
          "Treelite C API is required for predictions using unserialized "
          "rand_forest model!");

#endif
      });
  }
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
