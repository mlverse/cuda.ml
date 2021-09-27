#pragma once

#include <Rcpp.h>

#ifdef HAS_CUML

namespace cuml4r {

Rcpp::List pca_fit_transform(Rcpp::NumericMatrix const& x, double const tol,
                             int const n_iters, int const verbosity,
                             int const n_components, int const algo,
                             bool const whiten, bool const transform_input);

Rcpp::NumericMatrix pca_inverse_transform(Rcpp::List model,
                                          Rcpp::NumericMatrix const& x);

Rcpp::List pca_get_state(Rcpp::List const& model);

Rcpp::List pca_set_state(Rcpp::List const& model_state);

}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
