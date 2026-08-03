#pragma once

#include <treelite/c_api.h>

#include <stdexcept>
#include <string>

namespace cuml4r {

class TreeliteHandle {
 public:
  explicit TreeliteHandle(TreeliteModelHandle const handle = nullptr) noexcept
    : handle_(handle) {}

  TreeliteHandle(TreeliteHandle const&) = delete;
  TreeliteHandle& operator=(TreeliteHandle const&) = delete;

  TreeliteHandle(TreeliteHandle&& other) noexcept
    : handle_(other.release()) {}

  ~TreeliteHandle() noexcept { reset(); }

  TreeliteHandle& operator=(TreeliteHandle&& other) noexcept {
    if (this != &other) {
      reset(other.release());
    }
    return *this;
  }

  void reset(TreeliteModelHandle const handle = nullptr) noexcept {
    if (handle_ != nullptr) {
      TreeliteFreeModel(handle_);
    }
    handle_ = handle;
  }

  TreeliteModelHandle release() noexcept {
    auto const handle = handle_;
    handle_ = nullptr;
    return handle;
  }

  bool empty() const noexcept { return handle_ == nullptr; }

  TreeliteModelHandle* out() noexcept {
    reset();
    return &handle_;
  }

  TreeliteModelHandle get() const noexcept { return handle_; }

 private:
  TreeliteModelHandle handle_;
};

inline void treelite_check(int const status, std::string const& context) {
  if (status != 0) {
    auto const* error = TreeliteGetLastError();
    throw std::runtime_error(context + ": " +
                             (error == nullptr ? "unknown Treelite error"
                                               : error));
  }
}

}  // namespace cuml4r
