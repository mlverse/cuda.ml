#include "cd_fit_impl.h"
#include "lm.h"
#include <functional>

namespace cuml4r {

__host__ Rcpp::List cd_fit(Rcpp::NumericMatrix const& x,
                           Rcpp::NumericVector const& y,
                           bool const fit_intercept, int const epochs,
                           int const loss, double const alpha,
                           double const l1_ratio, bool const shuffle,
                           double const tol) {
  using namespace std::placeholders;

  return lm_fit(x, y, fit_intercept,
                /*fit_impl=*/
                std::bind(detail::cd_fit_impl, _1, _2, epochs, loss, alpha,
                          l1_ratio, shuffle, tol));
}

}  // namespace cuml4r
