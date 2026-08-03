#pragma once

#include <vector>

namespace cuml4r {

template <typename T>
using pinned_host_vector = std::vector<T>;

}  // namespace cuml4r
