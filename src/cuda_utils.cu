#include "cuda_utils.h"

#include <Rcpp.h>

namespace cuml4r {

__host__ int currentDevice() {
  int dev_id;
  CUDA_RT_CALL(cudaGetDevice(&dev_id));
  return dev_id;
}

}  // namespace cuml4r
