#include "lm_params.h"

#include <cuml/solvers/solver.hpp>

namespace cuml4r {
namespace detail {

__host__ void sgd_fit_impl(raft::handle_t& handle, lm::Params const& params,
                           int const batch_size, int const epochs,
                           int const lr_type, double const eta0,
                           double const power_t, int const loss,
                           int const penalty, double const alpha,
                           double const l1_ratio, bool const shuffle,
                           double const tol, int const n_iter_no_change) {
  ML::Solver::sgdFit(handle,
                     /*input=*/params.d_input,
                     /*n_rows=*/params.n_rows,
                     /*n_cols=*/params.n_cols,
                     /*labels=*/params.d_labels,
                     /*coef=*/params.d_coef,
                     /*intercept=*/params.intercept,
                     /*fit_intercept=*/params.fit_intercept, batch_size, epochs,
                     lr_type, eta0, power_t, loss, penalty, alpha, l1_ratio,
                     shuffle, tol, n_iter_no_change);
}

}  // namespace detail
}  // namespace cuml4r
