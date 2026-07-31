#pragma once

#include <cuda_runtime.h>

namespace Rcpp {

template <typename... Args>
[[noreturn]] void stop(const char* fmt, Args&&... args);

}  // namespace Rcpp

#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                              \
  {                                                                     \
    auto const cudaStatus = (call);                                     \
    if (cudaSuccess != cudaStatus) {                                    \
      Rcpp::stop(                                                       \
        "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with " \
        "%s (%d).\n",                                                   \
        #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus),      \
        cudaStatus);                                                    \
    }                                                                   \
  }
#endif

namespace cuml4r {

int currentDevice();

}  // namespace cuml4r
