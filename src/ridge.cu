#include "lm.h"
#include "ridge_fit_impl.h"

namespace cuml4r {

__host__ Rcpp::List ridge_fit(Rcpp::NumericMatrix const& x,
                              Rcpp::NumericVector const& y,
                              bool const fit_intercept,
                              bool const normalize_input, double const alpha,
                              int const algo) {
  using namespace std::placeholders;

  return lm_fit(x, y, /*intercept_type=*/lm::InterceptType::HOST, fit_intercept,
                normalize_input,
                /*fit_impl=*/
                std::bind(detail::ridge_fit_impl, _1, _2, alpha, algo));
}

}  // namespace cuml4r
