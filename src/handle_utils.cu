#include "handle_utils.h"
#include "stream_allocator.h"

#ifdef HAS_CUML

#include <cuml/version_config.hpp>
#if CUML_VERSION_MAJOR >= 24
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_pool.hpp>

#include <memory>
#endif

namespace cuml4r {
namespace handle_utils {

__host__ void initializeHandle(raft::handle_t& handle,
                               rmm::cuda_stream_view stream_view,
                               std::size_t stream_pool_size) {
  if (stream_view.value() == 0) {
    stream_view = stream_allocator::getOrCreateStream();
  }
#if CUML_VERSION_MAJOR >= 24
  raft::resource::set_cuda_stream(handle, stream_view);
  raft::resource::set_cuda_stream_pool(
    handle, std::make_shared<rmm::cuda_stream_pool>(stream_pool_size));
#else
  static_cast<void>(stream_pool_size);
  handle.set_stream(stream_view.value());
#endif
}

}  // namespace handle_utils
}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
