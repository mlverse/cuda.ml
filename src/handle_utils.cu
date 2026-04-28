#include "handle_utils.h"
#include "stream_allocator.h"

#ifdef HAS_CUML

#include <cuml/version_config.hpp>
#include <raft/core/resource/cuda_stream.hpp>

namespace cuml4r {
namespace handle_utils {

__host__ void initializeHandle(raft::handle_t& handle,
                               rmm::cuda_stream_view stream_view) {
  if (stream_view.value() == 0) {
    stream_view = stream_allocator::getOrCreateStream();
  }
#if CUML_VERSION_MAJOR >= 24
  raft::resource::set_cuda_stream(handle, stream_view);
#else
  handle.set_stream(stream_view.value());
#endif
}

}  // namespace handle_utils
}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
