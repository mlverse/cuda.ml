#pragma once

#include <Rcpp.h>

#ifdef HAS_CUML

namespace cuml4r {

Rcpp::NumericMatrix tsne_fit(
  Rcpp::NumericMatrix const& x, int const dim, int const n_neighbors,
  float const theta, float const epssq, float const perplexity,
  int const perplexity_max_iter, float const perplexity_tol,
  float const early_exaggeration, float const late_exaggeration,
  int const exaggeration_iter, float const min_gain,
  float const pre_learning_rate, float const post_learning_rate,
  int const max_iter, float const min_grad_norm, float const pre_momentum,
  float const post_momentum, long long const random_state, int const verbosity,
  bool const initialize_embeddings, bool const square_distances,
  int const algo);

}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
