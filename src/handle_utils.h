#pragma once

#ifdef HAS_CUML

#include <cstddef>
#include <cuml/version_config.hpp>
#if CUML_VERSION_MAJOR >= 24
#include <raft/core/handle.hpp>
#else
#include <raft/handle.hpp>
#endif
#include <rmm/cuda_stream_view.hpp>

namespace cuml4r {
namespace handle_utils {

void initializeHandle(raft::handle_t& handle,
                      rmm::cuda_stream_view stream_view = {},
                      std::size_t stream_pool_size = 8);

}  // namespace handle_utils
}  // namespace cuml4r

#endif
