#include "lm.h"
#include "ols_fit_impl.h"
#include "preprocessor.h"

#include <cuml/version_config.hpp>

#include <functional>

namespace cuml4r {

__host__ Rcpp::List ols_fit(Rcpp::NumericMatrix const& x,
                            Rcpp::NumericVector const& y,
                            bool const fit_intercept,
                            bool const normalize_input, int const algo) {
  using namespace std::placeholders;

  return lm_fit(x, y,
#if (CUML4R_LIBCUML_VERSION(CUML_VERSION_MAJOR, CUML_VERSION_MINOR) >= \
     CUML4R_LIBCUML_VERSION(21, 10))
                /*intercept_type=*/lm::InterceptType::HOST,
#else
                /*intercept_type=*/
                fit_intercept ? lm::InterceptType::DEVICE
                              : lm::InterceptType::HOST,
#endif
                fit_intercept, normalize_input,
                /*fit_impl=*/std::bind(detail::ols_fit_impl, _1, _2, algo));
}

}  // namespace cuml4r
