#include "cd_fit_impl.h"
#include "lm.h"
#include "preprocessor.h"

#include <cuml/version_config.hpp>

#include <functional>

namespace cuml4r {

__host__ Rcpp::List cd_fit(Rcpp::NumericMatrix const& x,
                           Rcpp::NumericVector const& y,
                           bool const fit_intercept, bool const normalize_input,
                           int const epochs, int const loss, double const alpha,
                           double const l1_ratio, bool const shuffle,
                           double const tol) {
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
                /*fit_impl=*/
                std::bind(detail::cd_fit_impl, _1, _2, epochs, loss, alpha,
                          l1_ratio, shuffle, tol));
}

}  // namespace cuml4r
