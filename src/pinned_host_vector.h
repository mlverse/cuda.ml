#pragma once

#ifdef HAS_CUML

#include <cuml/version_config.hpp>
#include <thrust/host_vector.h>
#if CUML_VERSION_MAJOR >= 26
#include <cuda/memory_resource>
#include <thrust/mr/allocator.h>
#include <thrust/system/cuda/memory_resource.h>
#else
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#endif

#include <Rcpp.h>

namespace cuml4r {

#if CUML_VERSION_MAJOR >= 26
// CCCL 3.x removed pinned_allocator; use the new memory resource API
template <typename T>
using pinned_host_vector = thrust::host_vector<T>;
#else
template <typename T>
using pinned_host_vector =
  thrust::host_vector<T, thrust::cuda::experimental::pinned_allocator<T>>;
#endif

}  // namespace cuml4r

namespace Rcpp {
namespace traits {

template <template <class> class Container, typename T>
struct pinned_container_exporter {
  using type = RangeExporter<Container<T>>;
};

// enable range exporter for pinned_host_vector
template <typename T>
class Exporter<cuml4r::pinned_host_vector<T>>
  : public pinned_container_exporter<cuml4r::pinned_host_vector, T>::type {
 public:
  Exporter(SEXP x)
    : pinned_container_exporter<cuml4r::pinned_host_vector, T>::type(x) {}
};

}  // namespace traits
}  // namespace Rcpp

#else

#include "warn_cuml_missing.h"

#endif
