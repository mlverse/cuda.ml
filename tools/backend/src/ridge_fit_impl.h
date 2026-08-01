#pragma once

#include "lm_params.h"

namespace raft {

class handle_t;

}  // namespace raft

namespace cuml4r {
namespace detail {

void ridge_fit_impl(raft::handle_t&, lm::Params const& params,
                    double const alpha, int const algo);

}  // namespace detail
}  // namespace cuml4r
