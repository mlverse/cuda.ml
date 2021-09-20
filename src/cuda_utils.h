#pragma once

#ifdef HAS_CUML

#include <cuda_runtime.h>

#ifndef NORET
#if defined(__GNUC__) && __GNUC__ >= 3
#define NORET __attribute__((noreturn))
#else
#define NORET
#endif
#endif

#ifndef cudaEventWaitDefault
#define cudaEventWaitDefault 0x00
#endif

namespace Rcpp {

template <typename... Args>
void NORET stop(const char* fmt, Args&&... args);

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

#else

#include "warn_cuml_missing.h"

#endif
