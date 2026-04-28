#include "lm_params.h"
#include "preprocessor.h"

#include <cuml/solvers/solver.hpp>
#include <cuml/version_config.hpp>

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
#if (CUML4R_LIBCUML_VERSION(CUML_VERSION_MAJOR, CUML_VERSION_MINOR) < \
     CUML4R_LIBCUML_VERSION(24, 0))
                    /*normalize=*/params.normalize_input, epochs, loss, alpha,
#else
                    epochs,
#endif
                    loss, alpha, l1_ratio, shuffle, tol);
}

}  // namespace detail
}  // namespace cuml4r
