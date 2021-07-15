#pragma once

#if HAS_CUML

#include <thrust/host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

namespace cuml4r {

template <typename T>
using pinned_host_vector =
  thrust::host_vector<T, thrust::cuda::experimental::pinned_allocator<T>>;

}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
