#pragma once

#if HAS_CUML

#include "cuda_utils.h"

#include <memory>

namespace cuml4r {

struct unique_marker final {
 public:
  __host__ unique_marker() : handle_(nullptr, marker_deleter()) {
    cudaEvent_t e;
    CUDA_RT_CALL(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    handle_.reset(e);
  }

  unique_marker(unique_marker const&) = delete;
  unique_marker(unique_marker&&) = default;
  unique_marker& operator=(unique_marker const&) = delete;
  unique_marker& operator=(unique_marker&&) = default;

  ~unique_marker() = default;

  __host__ cudaEvent_t get() const noexcept { return handle_.get(); }

 private:
  struct marker_deleter final {
    __host__ void operator()(CUevent_st* e) const {
      if (e != nullptr) {
        cudaEventDestroy(e);
      }
    }
  };

  std::unique_ptr<CUevent_st, marker_deleter> handle_;
};

}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
