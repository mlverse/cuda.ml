#pragma once

#ifdef HAS_CUML

#include <cuml/version_config.hpp>

#if CUML_VERSION_MAJOR < 24

#include <memory>

namespace raft {
namespace mr {
namespace device {

class allocator;

}  // namespace device
}  // namespace mr
}  // namespace raft

namespace cuml4r {

std::shared_ptr<raft::mr::device::allocator> getDeviceAllocator();

}  // namespace cuml4r

#endif

#else

#include "warn_cuml_missing.h"

#endif
