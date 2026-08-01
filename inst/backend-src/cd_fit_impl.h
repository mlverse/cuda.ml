#pragma once

#include "lm_params.h"

namespace raft {

class handle_t;

}  // namespace raft

namespace cuml4r {
namespace detail {

void cd_fit_impl(raft::handle_t& handle, lm::Params const& params,
                 int const epochs, int const loss, double const alpha,
                 double const l1_ratio, bool const shuffle, double const tol);

}  // namespace detail
}  // namespace cuml4r
