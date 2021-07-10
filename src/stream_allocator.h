#pragma once

#include <memory>

namespace rmm {

class cuda_stream_view;

}  // namespace rmm

namespace cuml4r {
namespace stream_allocator {

/*
 * Utility functions ensuring at most one non-default CUDA stream is allocated
 * by `cuml4r` on the current device.
 *
 * If the user does not specify a CUDA stream explicitly when launching
 * an algorithm from `cuml4r`, then `cuml4r` will get from StreamAllocator a
 * non-default stream that is created on the current device and run the
 * algorithm using that stream.
 *
 */

// get or create a non-default stream on the current device
rmm::cuda_stream_view getOrCreateStream();

}  // namespace stream_allocator
}  // namespace cuml4r
