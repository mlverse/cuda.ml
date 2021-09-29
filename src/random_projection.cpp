#include "random_projection.h"

// [[Rcpp::export(".rproj_johnson_lindenstrauss_min_dim")]]
size_t rproj_johnson_lindenstrauss_min_dim(size_t const n_samples,
                                           double const eps) {
#ifdef HAS_CUML

  return cuml4r::rproj_johnson_lindenstrauss_min_dim(n_samples, eps);

#else

#include "warn_cuml_missing.h"

  // return a dummy value
  return 0;

#endif
}

// [[Rcpp::export(".rproj_fit")]]
SEXP rproj_fit(int const n_samples, int const n_features,
               int const n_components, double const eps,
               bool const gaussian_method, double const density,
               int const random_state) {
#ifdef HAS_CUML

  return cuml4r::rproj_fit(n_samples, n_features, n_components, eps,
                           gaussian_method, density, random_state);

#else

#include "warn_cuml_missing.h"

  return Rcpp::List();

#endif
}

// [[Rcpp::export(".rproj_transform")]]
Rcpp::NumericMatrix rproj_transform(SEXP rproj_ctx_xptr,
                                    Rcpp::NumericMatrix const& input) {
#ifdef HAS_CUML

  return cuml4r::rproj_transform(rproj_ctx_xptr, input);

#else

#include "warn_cuml_missing.h"

  // dummy values with distinct data points
  return Rcpp::NumericMatrix::diag(input.nrow(), 1.0);

#endif
}

// [[Rcpp::export(".rproj_get_state")]]
Rcpp::List rproj_get_state(SEXP model) {
#ifdef HAS_CUML

  return cuml4r::rproj_get_state(model);

#else

#include "warn_cuml_missing.h"

  return {};

#endif
}

// [[Rcpp::export(".rproj_set_state")]]
SEXP rproj_set_state(Rcpp::List const& model_state) {
#ifdef HAS_CUML

  return cuml4r::rproj_set_state(model_state);

#else

#include "warn_cuml_missing.h"

  return R_NilValue;

#endif
}
