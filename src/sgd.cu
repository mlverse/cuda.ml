#include "lm.h"
#include "sgd_fit_impl.h"

#include <functional>

namespace cuml4r {

__host__ Rcpp::List sgd_fit(
  Rcpp::NumericMatrix const& x, Rcpp::NumericVector const& y,
  bool const fit_intercept, int const batch_size, int const epochs,
  int const lr_type, double const eta0, double const power_t, int const loss,
  int const penalty, double const alpha, double const l1_ratio,
  bool const shuffle, double const tol, int const n_iter_no_change) {
  using namespace std::placeholders;

  return lm_fit(x, y,
                /*intercept_type=*/lm::InterceptType::HOST, fit_intercept,
                /*normalize_input=*/false,
                /*fit_impl=*/
                std::bind(detail::sgd_fit_impl, _1, _2, batch_size, epochs,
                          lr_type, eta0, power_t, loss, penalty, alpha,
                          l1_ratio, shuffle, tol, n_iter_no_change));
}

}  // namespace cuml4r
