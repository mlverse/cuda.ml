#pragma once

#ifdef HAS_CUML

#include "cuda_utils.h"
#include "preprocessor.h"
#include "unique_marker.cuh"

#include <thrust/async/copy.h>
#include <thrust/system/cuda/future.h>

#include <utility>

namespace cuml4r {

// perform a copy operation that is asynchronous with respect to the host
// and synchronous with respect to the stream specified
template <typename... Args>
__host__ CUML4R_NODISCARD auto async_copy(cudaStream_t stream, Args&&... args) {
  auto e = thrust::async::copy(std::forward<Args>(args)...);
  auto& s = e.stream();
  unique_marker m;
  CUDA_RT_CALL(cudaEventRecord(m.get(), s.get()));
  CUDA_RT_CALL(cudaStreamWaitEvent(stream, m.get(), cudaEventWaitDefault));
  return m;
}

}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
