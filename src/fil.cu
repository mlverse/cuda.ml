#include "async_utils.cuh"
#include "cuda_utils.h"
#include "fil_utils.h"
#include "handle_utils.h"
#include "matrix_utils.h"
#include "pinned_host_vector.h"
#include "preprocessor.h"
#include "stream_allocator.h"
#include "treelite_utils.cuh"

#include <cuml/fil/fil.h>
#include <thrust/async/copy.h>
#include <thrust/device_vector.h>
#include <treelite/c_api.h>

#include <Rcpp.h>

#include <memory>
#include <string>

namespace {

enum class ModelType { XGBoost, XGBoostJSON, LightGBM };

__host__ int treeliteLoadModel(ModelType const model_type, char const* filename,
                               cuml4r::TreeliteHandle& tl_handle) {
  switch (model_type) {
    case ModelType::XGBoost:
      return TreeliteLoadXGBoostModel(filename, tl_handle.get());
    case ModelType::XGBoostJSON:
      return TreeliteLoadXGBoostJSON(filename, tl_handle.get());
    case ModelType::LightGBM:
      return TreeliteLoadLightGBMModel(filename, tl_handle.get());
  }

  // unreachable
  return -1;
}

struct FILModel {
  __host__ FILModel(std::unique_ptr<raft::handle_t> handle,
                    cuml4r::fil::forest_uptr forest, size_t const num_classes)
    : handle_(std::move(handle)),
      forest_(std::move(forest)),
      numClasses_(num_classes) {}

  std::unique_ptr<raft::handle_t> const handle_;
  // NOTE: the destruction of `forest_` must precede the destruction of
  // `handle_`.
  cuml4r::fil::forest_uptr forest_;
  size_t const numClasses_;
};

}  // namespace

namespace cuml4r {

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

  ML::fil::treelite_params_t params;
  params.algo = static_cast<ML::fil::algo_t>(algo);
  params.output_class = classification;
  params.threshold = threshold;
  params.storage_type = static_cast<ML::fil::storage_type_t>(storage_type);
  params.blocks_per_sm = blocks_per_sm;
  params.threads_per_tree = threads_per_tree;
  params.n_items = n_items;
  params.pforest_shape_str = nullptr;

  auto stream_view = cuml4r::stream_allocator::getOrCreateStream();
  auto handle = std::make_unique<raft::handle_t>();
  cuml4r::handle_utils::initializeHandle(*handle, stream_view.value());

  auto forest = fil::make_forest(*handle, /*src=*/[&] {
    ML::fil::forest* f;
    ML::fil::from_treelite(/*handle=*/*handle, /*pforest=*/&f,
                           /*model=*/*tl_handle.get(),
                           /*tl_params=*/&params);
    return f;
  });

  size_t num_classes = 0;
  if (classification) {
    auto const rc = TreeliteQueryNumClass(/*handle=*/*tl_handle.get(),
                                          /*out=*/&num_classes);
    if (rc < 0) {
      char const* err = TreeliteGetLastError();
      Rcpp::stop("TreeliteQueryNumClass failed: %s.", err);
    }

    // Treelite returns 1 as number of classes for binary classification.
    num_classes = std::max(num_classes, size_t(2));
  }

  return Rcpp::XPtr<FILModel>(
    std::make_unique<FILModel>(
      /*handle=*/std::move(handle), std::move(forest), num_classes)
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
  auto const m = cuml4r::Matrix<float>(x, /*transpose=*/false);

  if (output_class_probabilities && model_xptr->numClasses_ == 0) {
    Rcpp::stop(
      "'output_class_probabilities' is not applicable for regressions!");
  }

  auto& handle = *(model_xptr->handle_);

  // ensemble input data
  auto const& h_x = m.values;
  thrust::device_vector<float> d_x(h_x.size());
  auto CUML4R_ANONYMOUS_VARIABLE(x_h2d) = cuml4r::async_copy(
    handle.get_stream(), h_x.cbegin(), h_x.cend(), d_x.begin());

  // ensemble output
  thrust::device_vector<float> d_preds(output_class_probabilities
                                         ? model_xptr->numClasses_ * m.numRows
                                         : m.numRows);

  ML::fil::predict(/*h=*/handle, /*f=*/model_xptr->forest_.get(),
                   /*preds=*/d_preds.data().get(),
                   /*data=*/d_x.data().get(), /*num_rows=*/m.numRows,
                   /*predict_proba=*/output_class_probabilities);

  cuml4r::pinned_host_vector<float> h_preds(d_preds.size());
  auto CUML4R_ANONYMOUS_VARIABLE(preds_d2h) = cuml4r::async_copy(
    handle.get_stream(), d_preds.cbegin(), d_preds.cend(), h_preds.begin());

  CUDA_RT_CALL(cudaStreamSynchronize(handle.get_stream()));

  return Rcpp::transpose(Rcpp::NumericMatrix(
    output_class_probabilities ? model_xptr->numClasses_ : 1, m.numRows,
    h_preds.begin()));
}

}  // namespace cuml4r
