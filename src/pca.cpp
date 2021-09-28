#include "pca.h"

// [[Rcpp::export(".pca_fit_transform")]]
Rcpp::List pca_fit_transform(Rcpp::NumericMatrix const& x, double const tol,
                             int const n_iters, int const verbosity,
                             int const n_components, int const algo,
                             bool const whiten, bool const transform_input) {
#ifdef HAS_CUML

  return cuml4r::pca_fit_transform(x, tol, n_iters, verbosity, n_components,
                                   algo, whiten, transform_input);

#else

#include "warn_cuml_missing.h"

  return {};

#endif
}

// [[Rcpp::export(".pca_inverse_transform")]]
Rcpp::NumericMatrix pca_inverse_transform(Rcpp::List model,
                                          Rcpp::NumericMatrix const& x) {
#ifdef HAS_CUML

  return cuml4r::pca_inverse_transform(model, x);

#else

#include "warn_cuml_missing.h"

  return {};

#endif
}

// [[Rcpp::export(".pca_get_state")]]
Rcpp::List pca_get_state(Rcpp::List const& model) {
#ifdef HAS_CUML

  return cuml4r::pca_get_state(model);

#else

#include "warn_cuml_missing.h"

  return {};

#endif
}

// [[Rcpp::export(".pca_set_state")]]
Rcpp::List pca_set_state(Rcpp::List const& model_state) {
#ifdef HAS_CUML

  return cuml4r::pca_set_state(model_state);

#else

#include "warn_cuml_missing.h"

  return {};

#endif
}
