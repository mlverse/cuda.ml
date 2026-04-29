#include "async_utils.cuh"
#include "cuda_utils.h"
#include "fil_utils.h"
#include "handle_utils.h"
#include "matrix_utils.h"
#include "pinned_host_vector.h"
#include "preprocessor.h"
#include "stream_allocator.h"
#include "treelite_utils.cuh"

#ifndef CUML4R_TREELITE_C_API_MISSING

#include <thrust/device_vector.h>
#include <treelite/c_api.h>
#include <treelite/tree.h>

#include <Rcpp.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace cuml4r {
namespace {

enum class ModelType { XGBoost, XGBoostJSON, LightGBM };

struct FILModel {
  __host__ FILModel(std::unique_ptr<raft::handle_t> handle,
                    fil::forest_uptr forest, bool const classification,
                    float const threshold, size_t const num_classes)
    : handle_(std::move(handle)),
      forest_(std::move(forest)),
      classification_(classification),
      threshold_(threshold),
      numClasses_(num_classes) {}

  std::unique_ptr<raft::handle_t> const handle_;
  // NOTE: the destruction of `forest_` must precede the destruction of
  // `handle_`.
  fil::forest_uptr forest_;
  bool const classification_;
  float const threshold_;
  size_t const numClasses_;
};

__host__ int treeliteLoadModel(ModelType const model_type, char const* filename,
                               TreeliteHandle& tl_handle) {
  auto constexpr config = "{}";
  switch (model_type) {
    case ModelType::XGBoost:
      return TreeliteLoadXGBoostModelLegacyBinary(filename, config,
                                                  tl_handle.get());
    case ModelType::XGBoostJSON:
      return TreeliteLoadXGBoostModelJSON(filename, config, tl_handle.get());
    case ModelType::LightGBM:
      return TreeliteLoadLightGBMModel(filename, config, tl_handle.get());
  }

  // unreachable
  return -1;
}

__host__ size_t treelite_num_classes(TreeliteHandle const& tl_handle,
                                     bool const classification) {
  if (!classification) {
    return 0;
  }

  auto const* model = static_cast<treelite::Model const*>(tl_handle.handle());
  auto num_classes =
    model->num_class.Size() > 0 ? static_cast<size_t>(model->num_class[0]) : 0;

  // Treelite uses one output for binary classification in some import paths.
  return std::max(num_classes, size_t(2));
}

template <typename F>
__host__ Rcpp::NumericMatrix make_matrix(size_t const n_rows,
                                         size_t const n_cols, F&& getter) {
  Rcpp::NumericMatrix out(n_rows, n_cols);
  for (size_t i = 0; i < n_rows; ++i) {
    for (size_t j = 0; j < n_cols; ++j) {
      out(i, j) = getter(i, j);
    }
  }
  return out;
}

}  // namespace

__host__ SEXP fil_load_model(int const model_type, std::string const& filename,
                             int const algo, bool const classification,
                             float const threshold, int const storage_type,
                             int const blocks_per_sm,
                             int const threads_per_tree, int const n_items) {
  Rcpp::List model;

  TreeliteHandle tl_handle;
  {
    auto const rc = treeliteLoadModel(
      /*model_type=*/static_cast<ModelType>(model_type),
      /*filename=*/filename.c_str(), tl_handle);
    if (rc < 0) {
      char const* err = TreeliteGetLastError();
      Rcpp::stop("Failed to load XGBoost model file '%s': %s.",
                 filename.c_str(), err);
    }
  }

  auto stream_view = stream_allocator::getOrCreateStream();
  auto handle = std::make_unique<raft::handle_t>();
  handle_utils::initializeHandle(*handle, stream_view.value());

  auto forest = fil::import_from_treelite(
    *handle, tl_handle, fil::tree_layout_from_storage_type(storage_type));
  auto const num_classes = treelite_num_classes(tl_handle, classification);

  return Rcpp::XPtr<FILModel>(
    std::make_unique<FILModel>(
      /*handle=*/std::move(handle), std::move(forest), classification,
      threshold, num_classes)
      .release());
}

__host__ int fil_get_num_classes(SEXP const& model) {
  auto const model_xptr = Rcpp::XPtr<FILModel>(model);
  return model_xptr->numClasses_;
}

__host__ Rcpp::NumericMatrix fil_predict(
  SEXP const& model, Rcpp::NumericMatrix const& x,
  bool const output_class_probabilities) {
  auto const model_xptr = Rcpp::XPtr<FILModel>(model);
  auto const m = Matrix<float>(x, /*transpose=*/false);

  if (output_class_probabilities && !model_xptr->classification_) {
    Rcpp::stop(
      "'output_class_probabilities' is not applicable for regressions!");
  }

  auto& handle = *(model_xptr->handle_);

  // ensemble input data
  auto const& h_x = m.values;
  thrust::device_vector<float> d_x(h_x.size());
  auto CUML4R_ANONYMOUS_VARIABLE(x_h2d) =
    async_copy(handle.get_stream(), h_x.cbegin(), h_x.cend(), d_x.begin());

  auto const n_outputs =
    static_cast<size_t>(model_xptr->forest_->num_outputs());
  thrust::device_vector<float> d_preds(n_outputs * m.numRows);

  fil::predict(handle, *model_xptr->forest_, d_preds.data().get(),
               d_x.data().get(), m.numRows);

  pinned_host_vector<float> h_preds(d_preds.size());
  auto CUML4R_ANONYMOUS_VARIABLE(preds_d2h) = async_copy(
    handle.get_stream(), d_preds.cbegin(), d_preds.cend(), h_preds.begin());

  CUDA_RT_CALL(cudaStreamSynchronize(handle.get_stream()));

  if (!model_xptr->classification_) {
    return make_matrix(m.numRows, n_outputs, [&](size_t const i,
                                                 size_t const j) {
      return h_preds[i * n_outputs + j];
    });
  }

  if (output_class_probabilities) {
    if (n_outputs == model_xptr->numClasses_) {
      return make_matrix(m.numRows, n_outputs, [&](size_t const i,
                                                   size_t const j) {
        return h_preds[i * n_outputs + j];
      });
    }
    if (n_outputs == 1 && model_xptr->numClasses_ == 2) {
      return make_matrix(m.numRows, 2, [&](size_t const i, size_t const j) {
        auto const p1 = static_cast<double>(h_preds[i]);
        return j == 0 ? 1.0 - p1 : p1;
      });
    }
    Rcpp::stop("FIL model returned %d outputs, but %d classes were expected.",
               static_cast<int>(n_outputs),
               static_cast<int>(model_xptr->numClasses_));
  }

  return make_matrix(m.numRows, 1, [&](size_t const i, size_t) {
    if (n_outputs == 1) {
      return model_xptr->numClasses_ == 2
               ? static_cast<double>(h_preds[i] >= model_xptr->threshold_)
               : static_cast<double>(h_preds[i]);
    }
    if (model_xptr->forest_->row_postprocessing() == ML::fil::row_op::max_index) {
      return static_cast<double>(h_preds[i * n_outputs]);
    }

    auto const row_begin = h_preds.begin() + i * n_outputs;
    return static_cast<double>(
      std::distance(row_begin, std::max_element(row_begin, row_begin + n_outputs)));
  });
}

}  // namespace cuml4r

#endif
