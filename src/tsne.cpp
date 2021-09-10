#include "tsne.h"

// [[Rcpp::export(".tsne_fit")]]
Rcpp::NumericMatrix tsne_fit(
  Rcpp::NumericMatrix const& x, int const dim, int const n_neighbors,
  float const theta, float const epssq, float const perplexity,
  int const perplexity_max_iter, float const perplexity_tol,
  float const early_exaggeration, float const late_exaggeration,
  int const exaggeration_iter, float const min_gain,
  float const pre_learning_rate, float const post_learning_rate,
  int const max_iter, float const min_grad_norm, float const pre_momentum,
  float const post_momentum, int64_t const random_state, int const verbosity,
  bool const initialize_embeddings, bool const square_distances,
  int const algo) {
#ifdef HAS_CUML

  return cuml4r::tsne_fit(
    x, dim, n_neighbors, theta, epssq, perplexity, perplexity_max_iter,
    perplexity_tol, early_exaggeration, late_exaggeration, exaggeration_iter,
    min_gain, pre_learning_rate, post_learning_rate, max_iter, min_grad_norm,
    pre_momentum, post_momentum, random_state, verbosity, initialize_embeddings,
    square_distances, algo);

#else

#include "warn_cuml_missing.h"

  // dummy values with distinct data points
  return Rcpp::NumericMatrix::diag(x.nrow(), 1);

#endif
}
