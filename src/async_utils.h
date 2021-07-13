#pragma once

#if HAS_CUML

#include "cuda_utils.h"
#include "preprocessor.h"
#include "unique_marker.h"

#include <thrust/async/copy.h>
#include <thrust/system/cuda/future.h>

#include <utility>

namespace cuml4r {

// perform a copy operation that is asynchronous with respect to the host
// and synchronous with respect to the stream specified
template <typename... Args>
CUML4R_NODISCARD auto async_copy(cudaStream_t stream, Args&&... args) {
  auto s = thrust::async::copy(std::forward<Args>(args)...);
  unique_marker m;
  CUDA_RT_CALL(cudaStreamWaitEvent(stream, m.get(), 0));
  return m;
}

}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
