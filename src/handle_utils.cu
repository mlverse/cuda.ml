#include "handle_utils.h"
#include "stream_allocator.h"

#ifdef HAS_CUML

namespace cuml4r {
namespace handle_utils {

__host__ void initializeHandle(raft::handle_t& handle,
                               rmm::cuda_stream_view stream_view) {
  if (stream_view.value() == 0) {
    stream_view = stream_allocator::getOrCreateStream();
  }
  handle.set_stream(stream_view.value());
}

}  // namespace handle_utils
}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
