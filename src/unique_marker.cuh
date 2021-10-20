#pragma once

#ifdef HAS_CUML

#include "cuda_utils.h"

#include <memory>

namespace cuml4r {

struct unique_marker final {
 public:
  __host__ unique_marker() : event_(nullptr, marker_deleter()) {
    cudaEvent_t event;
    CUDA_RT_CALL(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    event_.reset(event);
  }

  unique_marker(unique_marker const&) = delete;
  unique_marker(unique_marker&&) = default;
  unique_marker& operator=(unique_marker const&) = delete;
  unique_marker& operator=(unique_marker&&) = default;

  ~unique_marker() = default;

  __host__ cudaEvent_t get() const noexcept { return event_.get(); }

 private:
  struct marker_deleter final {
    __host__ void operator()(CUevent_st* e) const {
      if (e != nullptr) {
        cudaEventDestroy(e);
      }
    }
  };

  std::unique_ptr<CUevent_st, marker_deleter> event_;
};

}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
