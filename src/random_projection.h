#pragma once

#include <Rcpp.h>

#ifdef HAS_CUML

namespace cuml4r {

size_t rproj_johnson_lindenstrauss_min_dim(size_t const n_samples,
                                           double const eps);

SEXP rproj_fit(int const n_samples, int const n_features,
               int const n_components, double const eps,
               bool const gaussian_method, double const density,
               int const random_state);

Rcpp::NumericMatrix rproj_transform(SEXP rproj_ctx_xptr,
                                    Rcpp::NumericMatrix const& input);

Rcpp::List rproj_get_state(SEXP model);

SEXP rproj_set_state(Rcpp::List const& model_state);

}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
