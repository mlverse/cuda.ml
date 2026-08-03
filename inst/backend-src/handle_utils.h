#pragma once

#include <cstddef>
#include <cstdint>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>

namespace cuml4r {
namespace handle_utils {

void initializeHandle(raft::handle_t& handle,
                      rmm::cuda_stream_view stream_view = {},
                      std::size_t stream_pool_size = 8);

}  // namespace handle_utils
}  // namespace cuml4r
