#include "lm.h"
#include "ols_fit_impl.h"
#include <functional>

namespace cuml4r {

__host__ Rcpp::List ols_fit(Rcpp::NumericMatrix const& x,
                            Rcpp::NumericVector const& y,
                            bool const fit_intercept, int const algo) {
  using namespace std::placeholders;

  return lm_fit(x, y, fit_intercept,
                /*fit_impl=*/std::bind(detail::ols_fit_impl, _1, _2, algo));
}

}  // namespace cuml4r
