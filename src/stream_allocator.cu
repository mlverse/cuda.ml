#ifdef HAS_CUML

#include "cuda_utils.h"
#include "device_allocator.h"
#include "stream_allocator.h"

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cstdlib>
#include <memory>
#include <unordered_map>

namespace {

using CudaStreamsMap = std::unordered_map<int, rmm::cuda_stream>;

auto const gCudaStreamsMap = std::make_shared<CudaStreamsMap>();

__host__ void destroyCudaStreams() { gCudaStreamsMap->clear(); }

__host__ std::shared_ptr<CudaStreamsMap> registerCudaStreamsMap() {
  // destroy all CUDA streams created by `cuml4r` before CUDA driver shuts down
  std::atexit(destroyCudaStreams);
  return gCudaStreamsMap;
}

__host__ CudaStreamsMap& cudaStreamsMap() {
  static auto streams = registerCudaStreamsMap();
  return *streams;
}

}  // namespace

namespace cuml4r {
namespace stream_allocator {

__host__ rmm::cuda_stream_view getOrCreateStream() {
  auto const dev_id = currentDevice();
  auto& cuda_streams_map = cudaStreamsMap();
  auto it = cuda_streams_map.find(dev_id);
  if (it != cuda_streams_map.end()) {
    return it->second.value();
  }
  auto const device_allocator = getDeviceAllocator();
  auto stream = rmm::cuda_stream();
  auto stream_view = stream.view();
  cudaStreamsMap().emplace(dev_id, std::move(stream));
  return stream_view;
}

}  // namespace stream_allocator
}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
