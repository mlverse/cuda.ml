#include "lm_params.h"

#include <cuml/solvers/solver.hpp>

namespace cuml4r {
namespace detail {

__host__ void cd_fit_impl(raft::handle_t& handle, lm::Params const& params,
                          int const epochs, int const loss, double const alpha,
                          double const l1_ratio, bool const shuffle,
                          double const tol) {
  ML::Solver::cdFit(handle, /*input=*/params.d_input,
                    /*n_rows=*/params.n_rows, /*n_cols=*/params.n_cols,
                    /*labels=*/params.d_labels, /*coef=*/params.d_coef,
                    /*intercept=*/params.intercept,
                    /*fit_intercept=*/params.fit_intercept,
                    /*normalize=*/params.normalize_input, epochs, loss, alpha,
                    l1_ratio, shuffle, tol);
}

}  // namespace detail
}  // namespace cuml4r
