#pragma once

#ifdef HAS_CUML

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

#else

#include "warn_cuml_missing.h"

#endif
