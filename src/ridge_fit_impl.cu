#include "lm_params.h"

#include <cuml/linear_model/glm.hpp>

namespace cuml4r {
namespace detail {

__host__ void ridge_fit_impl(raft::handle_t& handle, lm::Params const& params,
                             double const alpha, int const algo) {
  ML::GLM::ridgeFit(handle, /*input=*/params.d_input,
                    /*n_rows=*/params.n_rows,
                    /*n_cols=*/params.n_cols,
                    /*labels=*/params.d_labels,
                    /*alpha=*/const_cast<double*>(&alpha),
                    /*n_alpha=*/1,
                    /*coef=*/params.d_coef,
                    /*intercept=*/params.intercept,
                    /*fit_intercept=*/params.fit_intercept,
                    /*normalize=*/params.normalize_input, algo);
}

}  // namespace detail
}  // namespace cuml4r
