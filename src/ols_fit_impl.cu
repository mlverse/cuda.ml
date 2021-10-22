#include "lm_params.h"

#include <cuml/linear_model/glm.hpp>

namespace cuml4r {
namespace detail {

__host__ void ols_fit_impl(raft::handle_t& handle, lm::Params const& params,
                           int const algo) {
  ML::GLM::olsFit(handle, /*input=*/params.d_input,
                  /*n_rows=*/params.n_rows,
                  /*n_cols=*/params.n_cols,
                  /*labels=*/params.d_labels,
                  /*coef=*/params.d_coef,
                  /*intercept=*/params.intercept,
                  /*fit_intercept=*/params.fit_intercept,
                  /*normalize=*/params.normalize_input, algo);
}

}  // namespace detail
}  // namespace cuml4r
