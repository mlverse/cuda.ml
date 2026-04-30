#pragma once

#ifdef HAS_CUML

#include "cuda_utils.h"
#include "preprocessor.h"

#include <thrust/copy.h>
#include <thrust/system/cuda/execution_policy.h>

namespace cuml4r {

struct AsyncCopyCtx {};

// perform a copy operation that is asynchronous with respect to the host
// and synchronous with respect to the stream specified
template <typename InputIt, typename OutputIt>
__host__ CUML4R_NODISCARD auto async_copy(
  cudaStream_t stream, InputIt first, InputIt last, OutputIt result) {
  thrust::copy(thrust::cuda::par.on(stream), first, last, result);
  return AsyncCopyCtx{};
}

}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
