#pragma once

#if HAS_CUML

#include <cuda_runtime.h>
#include <Rcpp.h>

#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                              \
  {                                                                     \
    auto const cudaStatus = (call);                                     \
    if (cudaSuccess != cudaStatus) {                                    \
      Rcpp::stop(                                                       \
        "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with " \
        "%s (%d).\n",                                                   \
        #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus),      \
        cudaStatus                                                      \
      );                                                                \
    }                                                                   \
  }
#endif

#else
#warning `cuml4r` requires a valid RAPIDS installation. Please follow https://rapids.ai/start.html to install RAPIDS first. `cuml4r` must be installed and run from an environment containing a valid CUDA_PATH env variable (e.g., '/home/user/anaconda3/envs/rapids-21.06' or similar).
#endif
