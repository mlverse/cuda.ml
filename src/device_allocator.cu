#ifdef HAS_CUML

#include "device_allocator.h"

#include <cuml/version_config.hpp>

#if CUML_VERSION_MAJOR < 24

#include <raft/mr/device/allocator.hpp>

namespace {

auto const kDefaultDeviceAllocator =
  std::make_shared<raft::mr::device::default_allocator>();

}  // namespace

namespace cuml4r {

__host__ std::shared_ptr<raft::mr::device::allocator> getDeviceAllocator() {
  return kDefaultDeviceAllocator;
}

}  // namespace cuml4r

#endif

#else

#include "warn_cuml_missing.h"

#endif
