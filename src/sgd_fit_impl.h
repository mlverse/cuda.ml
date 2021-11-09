#pragma once

#include "lm_params.h"

namespace raft {

class handle_t;

}  // namespace raft

namespace cuml4r {
namespace detail {

void sgd_fit_impl(raft::handle_t& handle, lm::Params const& params,
                  int const batch_size, int const epochs, int const lr_type,
                  double const eta0, double const power_t, int const loss,
                  int const penalty, double const alpha, double const l1_ratio,
                  bool const shuffle, double const tol,
                  int const n_iter_no_change);

}  // namespace detail
}  // namespace cuml4r
