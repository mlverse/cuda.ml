#pragma once

#ifdef HAS_CUML

#include <vector>

namespace cuml4r {

template <typename T>
using pinned_host_vector = std::vector<T>;

}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
