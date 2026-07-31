#pragma once

#include "cuda_utils.h"
#include "preprocessor.h"

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/iterator/iterator_traits.h>

#include <iterator>
#include <type_traits>

namespace cuml4r {

struct AsyncCopyCtx {};

namespace detail {

inline cudaMemcpyKind copy_kind(thrust::host_system_tag,
                                thrust::device_system_tag) {
  return cudaMemcpyHostToDevice;
}

inline cudaMemcpyKind copy_kind(thrust::device_system_tag,
                                thrust::host_system_tag) {
  return cudaMemcpyDeviceToHost;
}

inline cudaMemcpyKind copy_kind(thrust::device_system_tag,
                                thrust::device_system_tag) {
  return cudaMemcpyDeviceToDevice;
}

}  // namespace detail

// perform a copy operation that is asynchronous with respect to the host
// and synchronous with respect to the stream specified
template <typename InputIt, typename OutputIt>
__host__ CUML4R_NODISCARD auto async_copy(cudaStream_t stream, InputIt first,
                                          InputIt last, OutputIt result) {
  using InputValue = typename thrust::iterator_traits<InputIt>::value_type;
  using OutputValue = typename thrust::iterator_traits<OutputIt>::value_type;
  using InputSystem = typename thrust::iterator_system<InputIt>::type;
  using OutputSystem = typename thrust::iterator_system<OutputIt>::type;

  static_assert(std::is_same<InputValue, OutputValue>::value,
                "async_copy requires matching input and output value types");

  auto const count = std::distance(first, last);
  if (count > 0) {
    CUDA_RT_CALL(cudaMemcpyAsync(
      thrust::raw_pointer_cast(&*result), thrust::raw_pointer_cast(&*first),
      sizeof(InputValue) * count,
      detail::copy_kind(InputSystem{}, OutputSystem{}), stream));
  }
  return AsyncCopyCtx{};
}

}  // namespace cuml4r
