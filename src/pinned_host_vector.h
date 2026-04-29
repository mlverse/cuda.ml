#pragma once

#ifdef HAS_CUML

#include <thrust/host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

#include <Rcpp.h>

namespace cuml4r {

template <typename T>
using pinned_host_vector =
  thrust::host_vector<T, thrust::cuda::experimental::pinned_allocator<T>>;

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
