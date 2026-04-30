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
#include <cuml/tree/decisiontree.hpp>
#ifndef CUML4R_TREELITE_C_API_MISSING
#include <cuml/fil/postproc_ops.hpp>
#endif
#include <cuml/version_config.hpp>

#include <Rcpp.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

namespace cuml4r {
namespace {

CUML4R_MAYBE_UNUSED constexpr auto kRfClassiferNumFeatures = "n_features";
CUML4R_MAYBE_UNUSED constexpr auto kRfClassifierForest = "forest";
CUML4R_MAYBE_UNUSED constexpr auto kRfClassifierInvLabelsMap = "inv_labels_map";

using RandomForestClassifierUPtr =
  std::unique_ptr<ML::RandomForestClassifierD,
                  RandomForestMetaDataDeleter<double, int>>;

class RandomForestClassifier {
 public:
  __host__ RandomForestClassifier(int const n_features,
                                  std::unordered_map<int, int>&& inv_labels_map,
                                  RandomForestClassifierUPtr rf) noexcept
    : nFeatures_(n_features),
      invLabelsMap_(std::move(inv_labels_map)),
      rf_(std::move(rf)) {}

#ifndef CUML4R_TREELITE_C_API_MISSING

  __host__ explicit RandomForestClassifier(Rcpp::List const& state)
    : nFeatures_(state[kRfClassiferNumFeatures]), rf_(nullptr) {
    {
      Rcpp::List inv_labels_map = state[kRfClassifierInvLabelsMap];
      for (auto const& p : inv_labels_map) {
        auto const kv = Rcpp::as<Rcpp::IntegerVector>(p);
        invLabelsMap_.emplace(kv[0], kv[1]);
      }
    }
    {
      std::unique_ptr<treelite::Model> model;
      std::vector<detail::PyBufFrameContent> py_buf_frames_content;
      detail::setState(model, py_buf_frames_content,
                       /*state=*/state[kRfClassifierForest]);
      tlHandle_ = model.release();
      pyBufFramesContent_ = std::move(py_buf_frames_content);
    }
  }

  __host__ Rcpp::List getState() {
    Rcpp::List state;

    state[kRfClassiferNumFeatures] = nFeatures_;
    {
      auto const& treelite_handle = getTreeliteHandle();

      state[kRfClassifierForest] = detail::getState(
        *static_cast<treelite::Model const*>(treelite_handle.handle()));
    }
    {
      Rcpp::List inv_labels_map;
      for (auto const& p : invLabelsMap_) {
        inv_labels_map.push_back(
          Rcpp::IntegerVector::create(p.first, p.second));
      }
      state[kRfClassifierInvLabelsMap] = std::move(inv_labels_map);
    }

    return state;
  }

#endif

  int const nFeatures_;
  std::unordered_map<int, int> invLabelsMap_;
  RandomForestClassifierUPtr const rf_;

#ifndef CUML4R_TREELITE_C_API_MISSING

 public:
  TreeliteHandle const& getTreeliteHandle() {
    if (tlHandle_.empty()) {
      tlHandle_ = detail::build_treelite_forest(
        /*forest=*/rf_.get(),
        /*n_features=*/nFeatures_, /*n_classes=*/invLabelsMap_.size());
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

// map labels into consecutive integral values in [0, to n_unique_labels)
__host__ pinned_host_vector<int> preprocess_labels(
  std::unordered_map<int, int>& labels_map, std::vector<int> const& labels) {
  int n_unique_labels = 0;
  pinned_host_vector<int> preprocessed_labels;
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
template <typename T>
__host__ void postprocess_labels(
  pinned_host_vector<T>& labels,
  std::unordered_map<int, int> const& inv_labels_map) {
  for (auto& label : labels) {
    auto iter = inv_labels_map.find(static_cast<int>(label));
    if (iter != inv_labels_map.cend()) {
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

template <typename InputT, typename OutputT>
__host__ Rcpp::IntegerVector rf_classifier_predict(
  Rcpp::XPtr<RandomForestClassifier> const& model,
  Rcpp::NumericMatrix const& input,
  std::function<void(raft::handle_t const&, InputT* const,
                     OutputT* const)> const& predict_impl) {
  auto const input_m = Matrix<InputT>(input, /*transpose=*/false);
  auto const n_samples = input_m.numRows;

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
  thrust::device_vector<OutputT> d_predictions(n_samples);

  predict_impl(handle, d_input.data().get(), d_predictions.data().get());

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  pinned_host_vector<OutputT> h_predictions(n_samples);
  AsyncCopyCtx __attribute__((unused)) preds_d2h;
  preds_d2h = async_copy(stream_view.value(), d_predictions.cbegin(),
                         d_predictions.cend(), h_predictions.begin());

  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  // post-process prediction labels
  postprocess_labels(h_predictions, model->invLabelsMap_);

  return Rcpp::IntegerVector(h_predictions.begin(), h_predictions.end());
}

}  // namespace

__host__ SEXP rf_classifier_fit(
  Rcpp::NumericMatrix const& input, Rcpp::IntegerVector const& labels,
  int const n_trees, bool const bootstrap, float const max_samples,
  int const n_streams, int const max_depth, int const max_leaves,
  float const max_features, int const n_bins, int const min_samples_leaf,
  int const min_samples_split, int const split_criterion,
  float const min_impurity_decrease, int const max_batch_size,
  int const verbosity) {
  auto const input_m = Matrix<>(input, /*transpose=*/true);
  int const n_samples = input_m.numCols;
  int const n_features = input_m.numRows;

  auto rf = RandomForestClassifierUPtr(new ML::RandomForestClassifierD);

  auto stream_view = stream_allocator::getOrCreateStream();
  raft::handle_t handle;
  handle_utils::initializeHandle(handle, stream_view.value());

  // rf input data & labels
  auto const& h_input = input_m.values;
  thrust::device_vector<double> d_input(h_input.size());
  auto CUML4R_ANONYMOUS_VARIABLE(input_h2d) = async_copy(
    stream_view.value(), h_input.cbegin(), h_input.cend(), d_input.begin());
  std::unordered_map<int, int> labels_map;
  auto const h_labels =
    preprocess_labels(labels_map, Rcpp::as<std::vector<int>>(labels));

  thrust::device_vector<int> d_labels(h_labels.size());
  auto CUML4R_ANONYMOUS_VARIABLE(labels_h2d) = async_copy(
    stream_view.value(), h_labels.cbegin(), h_labels.cend(), d_labels.begin());
  {
    auto* rf_ptr = rf.get();
    ML::fit(handle, rf_ptr, d_input.data().get(), n_samples, n_features,
            d_labels.data().get(),
            /*n_unique_labels=*/static_cast<int>(labels_map.size()),
#if (CUML4R_LIBCUML_VERSION(CUML_VERSION_MAJOR, CUML_VERSION_MINOR) >= \
     CUML4R_LIBCUML_VERSION(21, 8))

            ML::set_rf_params(
              max_depth, max_leaves, max_features, n_bins, min_samples_leaf,
              min_samples_split, min_impurity_decrease, bootstrap, n_trees,
              max_samples,
              /*seed=*/0,
              /*split_criterion=*/static_cast<ML::CRITERION>(split_criterion),
              /*cfg_n_strems=*/n_streams, max_batch_size),

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
      rf = RandomForestClassifierUPtr(rf_ptr);
    }
  }

  auto model = std::make_unique<RandomForestClassifier>(
    n_features, reverse(labels_map), std::move(rf));

  return Rcpp::XPtr<RandomForestClassifier>(model.release());
}

__host__ Rcpp::IntegerVector rf_classifier_predict(
  SEXP model_xptr, Rcpp::NumericMatrix const& input, int const verbosity) {
  int const n_samples = input.nrow();
  int const n_features = input.ncol();
  auto const model = Rcpp::XPtr<RandomForestClassifier>(model_xptr);

  if (model->rf_ != nullptr) {
    return rf_classifier_predict<double, int>(
      model, input,
      /*predict_impl=*/
      [n_samples, n_features, verbosity, rf = model->rf_.get()](
        raft::handle_t const& handle, double* const d_input,
        int* const d_preds) {
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
    auto const row_begin = h_raw_predictions.begin() + i * n_outputs;
    h_predictions[i] =
      forest->row_postprocessing() == ML::fil::row_op::max_index || n_outputs == 1
        ? row_begin[0]
        : static_cast<float>(
            std::distance(row_begin,
                          std::max_element(row_begin, row_begin + n_outputs)));
  }

  postprocess_labels(h_predictions, model->invLabelsMap_);

  return Rcpp::IntegerVector(h_predictions.begin(), h_predictions.end());
#else
  Rcpp::stop(
    "Treelite C API is required for predictions using unserialized "
    "rand_forest model!");

#endif
}

__host__ Rcpp::NumericMatrix rf_classifier_predict_class_probabilities(
  SEXP model_xptr, Rcpp::NumericMatrix const& input) {
#ifndef CUML4R_TREELITE_C_API_MISSING

  auto const input_m = Matrix<float>(input, /*transpose=*/false);
  int const n_samples = input_m.numRows;

  auto model = Rcpp::XPtr<RandomForestClassifier>(model_xptr);
  int const n_classes = model->invLabelsMap_.size();

  auto const& tl_handle = model->getTreeliteHandle();

  auto stream_view = stream_allocator::getOrCreateStream();
  raft::handle_t handle;
  handle_utils::initializeHandle(handle, stream_view.value());

  auto forest = fil::import_from_treelite(handle, tl_handle);
  forest->set_row_postprocessing(ML::fil::row_op::disable);
  auto const n_outputs = static_cast<size_t>(forest->num_outputs());

  // FIL input
  auto const& h_x = input_m.values;
  thrust::device_vector<float> d_x(h_x.size());
  auto CUML4R_ANONYMOUS_VARIABLE(x_h2d) =
    async_copy(handle.get_stream(), h_x.cbegin(), h_x.cend(), d_x.begin());

  // FIL output
  thrust::device_vector<float> d_preds(n_outputs * n_samples);

  fil::predict(handle, *forest, d_preds.data().get(), d_x.data().get(),
               n_samples);

  CUDA_RT_CALL(cudaStreamSynchronize(handle.get_stream()));

  pinned_host_vector<float> h_preds(d_preds.size());
  auto CUML4R_ANONYMOUS_VARIABLE(preds_d2h) = async_copy(
    handle.get_stream(), d_preds.cbegin(), d_preds.cend(), h_preds.begin());

  CUDA_RT_CALL(cudaStreamSynchronize(handle.get_stream()));

  if (n_outputs == static_cast<size_t>(n_classes)) {
    return Rcpp::transpose(
      Rcpp::NumericMatrix(n_outputs, n_samples, h_preds.begin()));
  }
  if (n_outputs == 1 && n_classes == 2) {
    Rcpp::NumericMatrix out(n_samples, 2);
    for (int i = 0; i < n_samples; ++i) {
      out(i, 1) = h_preds[i];
      out(i, 0) = 1.0 - out(i, 1);
    }
    return out;
  }
  Rcpp::stop("FIL model returned %d outputs, but %d classes were expected.",
             static_cast<int>(n_outputs), n_classes);
#else

  return {};

#endif
}

__host__ Rcpp::List rf_classifier_get_state(SEXP model) {
#ifndef CUML4R_TREELITE_C_API_MISSING

  return Rcpp::XPtr<RandomForestClassifier>(model)->getState();

#else

  return {};

#endif
}

__host__ SEXP rf_classifier_set_state(Rcpp::List const& state) {
#ifndef CUML4R_TREELITE_C_API_MISSING

  auto model = std::make_unique<RandomForestClassifier>(state);
  return Rcpp::XPtr<RandomForestClassifier>(model.release());

#else

  return R_NilValue;

#endif
}

}  // namespace cuml4r
