#pragma once

#ifdef HAS_CUML

#include <treelite/c_api.h>
#include <cuml/ensemble/randomforest.hpp>

#ifndef CUML4R_TREELITE_C_API_MISSING

#include <treelite/tree.h>

#endif

namespace cuml4r {

#ifndef CUML4R_TREELITE_C_API_MISSING

class TreeliteHandle {
 public:
  __host__ explicit TreeliteHandle(ModelHandle const handle = nullptr) noexcept
    : handle_(handle) {}

  __host__ TreeliteHandle(TreeliteHandle const& o) = delete;

  __host__ TreeliteHandle(TreeliteHandle&& o) noexcept
    : TreeliteHandle(o.handle_) {
    o.handle_ = nullptr;
  }

  __host__ ~TreeliteHandle() noexcept {
    if (handle_ != nullptr) {
      TreeliteFreeModel(handle_);
    }
  }

  __host__ TreeliteHandle& operator=(TreeliteHandle&& o) noexcept {
    if (handle_ != nullptr) {
      TreeliteFreeModel(handle_);
    }
    handle_ = o.handle_;
    o.handle_ = nullptr;
    return *this;
  }

  __host__ TreeliteHandle& operator=(ModelHandle const handle) noexcept {
    if (handle_ != nullptr) {
      TreeliteFreeModel(handle_);
    }
    handle_ = handle;
    return *this;
  }

  __host__ bool empty() const noexcept { return handle_ == nullptr; }

  __host__ ModelHandle const* get() const noexcept { return &handle_; }

  __host__ ModelHandle* get() noexcept { return &handle_; }

 private:
  ModelHandle handle_;
};

namespace detail {

template <typename T, typename L>
__host__ TreeliteHandle
build_treelite_forest(ML::RandomForestMetaData<T, L> const* forest,
                      int const n_features, int const n_classes) {
  TreeliteHandle handle;
  ML::build_treelite_forest(/*model=*/handle.get(), forest,
                            /*num_features=*/n_features,
                            /*task_category=*/n_classes);

  return handle;
}

}  // namespace detail

#endif

template <typename T, typename L>
struct RandomForestMetaDataDeleter {
  __host__ void operator()(ML::RandomForestMetaData<T, L>* const rf) const {
    ML::delete_rf_metadata<T, L>(rf);
  }
};

}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
