#include "tsvd.h"

// [[Rcpp::export(".tsvd_fit_transform")]]
Rcpp::List tsvd_fit_transform(Rcpp::NumericMatrix const& x, double const tol,
                              int const n_iters, int const verbosity,
                              int const n_components, int const algo,
                              bool const transform_input) {
#ifdef HAS_CUML

  return cuml4r::tsvd_fit_transform(x, tol, n_iters, verbosity, n_components,
                                    algo, transform_input);
#else

#include "warn_cuml_missing.h"

  return {};

#endif
}

// [[Rcpp::export(".tsvd_transform")]]
Rcpp::NumericMatrix tsvd_transform(Rcpp::List model,
                                   Rcpp::NumericMatrix const& x) {
#ifdef HAS_CUML

  return cuml4r::tsvd_transform(model, x);

#else

#include "warn_cuml_missing.h"

  return {};

#endif
}

// [[Rcpp::export(".tsvd_inverse_transform")]]
Rcpp::NumericMatrix tsvd_inverse_transform(Rcpp::List model,
                                           Rcpp::NumericMatrix const& x) {
#ifdef HAS_CUML

  return cuml4r::tsvd_inverse_transform(model, x);

#else

#include "warn_cuml_missing.h"

  return {};

#endif
}
