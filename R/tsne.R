tsne_match_method <- function(method = c("barnes_hut", "fft", "exact")) {
  method <- match.arg(method)

  switch(method,
    exact = 0L,
    barnes_hut = 1L,
    fft = 2L
  )
}

new_tsne_model <- function(embedding) {
  class(embedding) <- c("cuda_ml_tsne_model", "cuda_ml_model")

  embedding
}

#' t-distributed Stochastic Neighbor Embedding.
#'
#' t-distributed Stochastic Neighbor Embedding (TSNE) for visualizing high-
#' dimensional data.
#'
#' @template model-with-numeric-input
#' @template cuML-log-level
#' @param n_components Dimension of the embedded space.
#' @param n_neighbors The number of datapoints to use in the attractive forces.
#'   Default: ceiling(3 * perplexity).
#' @param method T-SNE method, must be one of {"barnes_hut", "fft", "exact"}.
#'   The "exact" method will be more accurate but slower. Both "barnes_hut" and
#'   "fft" methods are fast approximations.
#' @param angle Valid values are between 0.0 and 1.0, which trade off speed and
#'   accuracy, respectively. Generally, these values are set between 0.2 and
#'   0.8. (Barnes-Hut only.)
#' @param n_iter Maximum number of iterations for the optimization. Should be
#'   at least 250. Default: 1000L.
#' @param learning_rate Learning rate of the t-SNE algorithm, usually between
#'   (10, 1000). If the learning rate is too high, then t-SNE result could look
#'   like a cloud / ball of points.
#' @param learning_rate_method Must be one of {"adaptive", "none"}. If
#'   "adaptive", then learning rate, early exaggeration, and perplexity are
#'   automatically tuned based on input size. Default: "adaptive".
#' @param perplexity The target value of the conditional distribution's
#'   perplexity (see
#'   https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding
#'   for details).
#' @param perplexity_max_iter The number of epochs the best Gaussian bands are
#'   found for. Default: 100L.
#' @param perplexity_tol Stop optimizing the Gaussian bands when the conditional
#'   distribution's perplexity is within this desired tolerance compared to its
#'   taget value. Default: 1e-5.
#' @param early_exaggeration Controls the space between clusters. Not critical
#'   to tune this. Default: 12.0.
#' @param late_exaggeration Controls the space between clusters. It may be
#'   beneficial to increase this slightly to improve cluster separation. This
#'   will be applied after `exaggeration_iter` iterations (FFT only).
#' @param exaggeration_iter Number of exaggeration iterations. Default: 250L.
#' @param min_grad_norm If the gradient norm is below this threshold, the
#'   optimization will be stopped. Default: 1e-7.
#' @param pre_momentum During the exaggeration iteration, more forcefully apply
#'   gradients. Default: 0.5.
#' @param post_momentum During the late phases, less forcefully apply gradients.
#'   Default: 0.8.
#' @param square_distances Whether TSNE should square the distance values.
#' @param seed Seed to the psuedorandom number generator. Setting this can make
#'   repeated runs look more similar. Note, however, that this highly
#'   parallelized t-SNE implementation is not completely deterministic between
#'   runs, even with the same \code{seed} being used for each run.
#'   Default: NULL.
#'
#' @return A matrix containing the embedding of the input data in a low-
#'   dimensional space, with each row representing an embedded data point.
#'
#' @examples
#' library(cuda.ml)
#'
#' embedding <- cuda_ml_tsne(iris[1:4], method = "exact")
#'
#' set.seed(0L)
#' print(kmeans(embedding, centers = 3))
#' @export
cuda_ml_tsne <- function(x, n_components = 2L,
                         n_neighbors = ceiling(3 * perplexity),
                         method = c("barnes_hut", "fft", "exact"), angle = 0.5,
                         n_iter = 1000L, learning_rate = 200.0,
                         learning_rate_method = c("adaptive", "none"),
                         perplexity = 30.0, perplexity_max_iter = 100L,
                         perplexity_tol = 1e-5, early_exaggeration = 12.0,
                         late_exaggeration = 1.0, exaggeration_iter = 250L,
                         min_grad_norm = 1e-7, pre_momentum = 0.5,
                         post_momentum = 0.8, square_distances = TRUE, seed = NULL,
                         cuML_log_level = c("off", "critical", "error", "warn", "info", "debug", "trace")) {
  learning_rate_method <- match.arg(learning_rate_method)

  if (identical(learning_rate_method, "adaptive") &&
    method %in% c("barnes_hut", "fft")) {
    if (nrow(x) <= 2000L) {
      n_neighbors <- min(max(n_neighbors, 90L), nrow(x))
    } else {
      n_neighbors <- max(as.integer(102 - 0.0012 * nrow(x)), 30L)
    }

    pre_learning_rate <- max(nrow(x) / 3.0, 1.0)
    post_learning_rate <- pre_learning_rate
    early_exaggeration <- ifelse(nrow(x) > 10000L, 24.0, 12.0)
  } else {
    pre_learning_rate <- learning_rate
    post_learning_rate <- learning_rate * 2
  }
  algo <- tsne_match_method(method)
  cuML_log_level <- match_cuML_log_level(cuML_log_level)

  model_obj <- .tsne_fit(
    x = as.matrix(x),
    dim = as.integer(n_components),
    n_neighbors = as.integer(n_neighbors),
    algo = algo,
    initialize_embeddings = TRUE,
    square_distances = square_distances,
    theta = as.numeric(angle),
    epssq = 0.0025,
    perplexity = as.numeric(perplexity),
    perplexity_max_iter = as.integer(perplexity_max_iter),
    perplexity_tol = as.numeric(perplexity_tol),
    early_exaggeration = as.numeric(early_exaggeration),
    late_exaggeration = as.numeric(late_exaggeration),
    exaggeration_iter = as.integer(exaggeration_iter),
    min_gain = 1e-2,
    pre_learning_rate = as.numeric(pre_learning_rate),
    post_learning_rate = as.numeric(post_learning_rate),
    max_iter = as.integer(n_iter),
    min_grad_norm = as.numeric(min_grad_norm),
    pre_momentum = as.numeric(pre_momentum),
    post_momentum = as.numeric(post_momentum),
    random_state = as.integer(seed %||% -1L),
    verbosity = cuML_log_level
  )

  new_tsne_model(model_obj)
}
