#include "async_utils.cuh"
#include "cuda_utils.h"
#include "handle_utils.h"
#include "matrix_utils.h"
#include "pinned_host_vector.h"
#include "preprocessor.h"
#include "stream_allocator.h"

#include <thrust/async/copy.h>
#include <thrust/device_vector.h>
#include <cuml/fil/fil.h>
#include <treelite/c_api.h>

#include <Rcpp.h>

#include <memory>
#include <string>

namespace {

struct TreeliteModel {
  TreeliteModel(std::unique_ptr<raft::handle_t> handle,
                ML::fil::forest_t const forest, ModelHandle const model,
                size_t const num_classes)
    : handle_(std::move(handle)),
      forest_(forest),
      model_(model),
      numClasses_(num_classes) {}
  ~TreeliteModel() {
    if (forest_ != nullptr) {
      ML::fil::free(*handle_, forest_);
    }
  }

  std::unique_ptr<raft::handle_t> const handle_;
  ML::fil::forest_t const forest_;
  ModelHandle const model_;
  size_t const numClasses_;
};

}  // namespace

namespace cuml4r {

SEXP treelite_load_xgboost_model(
  std::string const& filename, int const algo, bool const output_class,
  float const threshold, int const storage_type, int const blocks_per_sm,
  int const threads_per_tree, int const n_items) {
  Rcpp::List model;

  ModelHandle model_handle;
  {
    auto const rc = TreeliteLoadXGBoostModel(/*filename=*/filename.c_str(),
                                             /*out=*/&model_handle);
    if (rc < 0) {
      char const* err = TreeliteGetLastError();
      Rcpp::stop("Failed to load XGBoost model file '%s': %s.",
                 filename.c_str(), err);
    }
  }

  ML::fil::treelite_params_t params;
  params.algo = static_cast<ML::fil::algo_t>(algo);
  params.output_class = output_class;
  params.threshold = threshold;
  params.storage_type = static_cast<ML::fil::storage_type_t>(storage_type);
  params.blocks_per_sm = blocks_per_sm;
  params.threads_per_tree = threads_per_tree;
  params.n_items = n_items;
  params.pforest_shape_str = nullptr;

  auto stream_view = cuml4r::stream_allocator::getOrCreateStream();
  auto handle = std::make_unique<raft::handle_t>();
  cuml4r::handle_utils::initializeHandle(*handle, stream_view.value());

  ML::fil::forest_t forest;

  ML::fil::from_treelite(/*handle=*/*handle, /*pforest=*/&forest,
                         /*model=*/model_handle, /*tl_params=*/&params);

  size_t num_classes = 0;
  {
    auto const rc =
      TreeliteQueryNumClass(/*handle=*/model_handle, /*out=*/&num_classes);
    if (rc < 0) {
      char const* err = TreeliteGetLastError();
      Rcpp::stop("TreeliteQueryNumClass failed: %s.", err);
    }
  }

  return Rcpp::XPtr<TreeliteModel>(std::make_unique<TreeliteModel>(
                                /*handle=*/std::move(handle), forest,
                                /*model=*/model_handle, num_classes)
                                .release());
}

Rcpp::NumericMatrix treelite_predict(
    SEXP const& model,
    Rcpp::NumericMatrix const& x,
    bool const output_probabilities) {
  auto const model_xptr = Rcpp::XPtr<TreeliteModel>(model);
  auto const m = cuml4r::Matrix<float>(x, /*transpose=*/false);

  auto& handle = *(model_xptr->handle_);

  // ensemble input data
  auto const& h_x = m.values;
  thrust::device_vector<float> d_x(h_x.size());
  auto CUML4R_ANONYMOUS_VARIABLE(x_h2d) =
    cuml4r::async_copy(handle.get_stream(), h_x.cbegin(), h_x.cend(), d_x.begin());

  // ensemble output
  thrust::device_vector<float> d_preds(output_probabilities ? 2 * m.numRows : m.numRows);

  ML::fil::predict(/*h=*/handle, /*f=*/model_xptr->forest_, /*preds=*/d_preds.data().get(),
    /*data=*/d_x.data().get(), /*num_rows=*/m.numRows, /*predict_proba=*/output_probabilities);

  cuml4r::pinned_host_vector<float> h_preds(d_preds.size());
  auto CUML4R_ANONYMOUS_VARIABLE(preds_d2h) =
    cuml4r::async_copy(handle.get_stream(), d_preds.cbegin(), d_preds.cend(), h_preds.begin());

  CUDA_RT_CALL(cudaStreamSynchronize(handle.get_stream()));

  return Rcpp::NumericMatrix(m.numRows, output_probabilities ? 2 : 1, h_preds.begin());
}

}  // namespace cuml4r
