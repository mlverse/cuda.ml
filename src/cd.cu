#include "cd_fit_impl.h"
#include "lm.h"

namespace cuml4r {

__host__ Rcpp::List cd_fit(Rcpp::NumericMatrix const& x,
                           Rcpp::NumericVector const& y,
                           bool const fit_intercept, bool const normalize_input,
                           int const epochs, int const loss, double const alpha,
                           double const l1_ratio, bool const shuffle,
                           double const tol) {
  using namespace std::placeholders;

  return lm_fit(x, y, /*intercept_type=*/
                        fit_intercept ? lm::InterceptType::DEVICE
                                      : lm::InterceptType::HOST,
                fit_intercept, normalize_input,
                /*fit_impl=*/
                std::bind(detail::cd_fit_impl, _1, _2, epochs, loss, alpha,
                          l1_ratio, shuffle, tol));
}

}  // namespace cuml4r
