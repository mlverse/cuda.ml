#pragma once

#include <treelite/c_api.h>

#ifndef CUML4R_TREELITE_C_API_MISSING

namespace cuml4r {

/*
 * RAII wrapper for the Treelite model handle.
 */
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

}  // namespace cuml4r

#endif
