#include "handle_utils.h"
#include "stream_allocator.h"

#ifdef HAS_CUML

#include <cuml/version_config.hpp>

namespace cuml4r {
namespace handle_utils {

__host__ void initializeHandle(raft::handle_t& handle,
                               rmm::cuda_stream_view stream_view) {
  if (stream_view.value() == 0) {
    stream_view = stream_allocator::getOrCreateStream();
  }
#if CUML_VERSION_MAJOR >= 25
  // In raft 26.x, handle_t takes stream_view in the constructor.
  // Reconstruct the handle with the desired stream via placement new.
  handle.~handle_t();
  new (&handle) raft::handle_t(stream_view);
#else
  handle.set_stream(stream_view.value());
#endif
}

}  // namespace handle_utils
}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
