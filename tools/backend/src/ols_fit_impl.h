#pragma once

#include "lm_params.h"

namespace raft {

class handle_t;

}  // namespace raft

namespace cuml4r {
namespace detail {

void ols_fit_impl(raft::handle_t&, lm::Params const& params, int const algo);

}  // namespace detail
}  // namespace cuml4r
