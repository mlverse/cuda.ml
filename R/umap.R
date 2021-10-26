umap_match_init_mode <- function(init = c("spectral", "random")) {
  init <- match.arg(init)

  switch(init,
    spectral = 1L,
    random = 0L
  )
}

umap_match_metric_type <- function(metric_type = c("categorical", "euclidean")) {
  metric_type <- match.arg(metric_type)

  switch(metric_type,
    categorical = 1L,
    euclidean = 0L
  )
}

new_umap_model <- function(model) {
  class(model) <- c("cuda_ml_umap", "cuda_ml_model", class(model))

  model
}

#' Uniform Manifold Approximation and Projection (UMAP) for dimension reduction.
#'
#' Run the Uniform Manifold Approximation and Projection (UMAP) algorithm to
#' find a low dimensional embedding of the input data that approximates an
#' underlying manifold.
#'
#' @template model-with-numeric-input
#' @template transform-input
#' @template cuML-log-level
#' @param y An optional numeric vector of target values for supervised dimension
#'   reduction. Default: NULL.
#' @param n_components The dimension of the space to embed into. Default: 2.
#' @param n_neighbors The size of local neighborhood (in terms of number of
#'   neighboring sample points) used for manifold approximation. Default: 15.
#' @param n_epochs The number of training epochs to be used in optimizing the
#'   low dimensional embedding. Default: 500.
#' @param learning_rate The initial learning rate for the embedding
#'   optimization. Default: 1.0.
#' @param init Initialization mode of the low dimensional embedding. Must be
#'   one of {"spectral", "random"}. Default: "spectral".
#' @param min_dist The effective minimum distance between embedded points.
#'   Default: 0.1.
#' @param spread The effective scale of embedded points. In combination with
#'   \code{min_dist} this determines how clustered/clumped the embedded points
#'   are. Default: 1.0.
#' @param set_op_mix_ratio Interpolate between (fuzzy) union and intersection as
#'   the set operation used to combine local fuzzy simplicial sets to obtain a
#'   global fuzzy simplicial sets. Both fuzzy set operations use the product
#'   t-norm. The value of this parameter should be between 0.0 and 1.0; a value
#'   of 1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
#'   intersection. Default: 1.0.
#' @param local_connectivity The local connectivity required -- i.e. the number
#'   of nearest neighbors that should be assumed to be connected at a local
#'   level. Default: 1.
#' @param repulsion_strength Weighting applied to negative samples in low
#'   dimensional embedding optimization. Values higher than one will result in
#'   greater weight being given to negative samples. Default: 1.0.
#' @param negative_sample_rate The number of negative samples to select per
#'   positive sample in the optimization process. Default: 5.
#' @param transform_queue_size For transform operations (embedding new points
#'   using a trained model this will control how aggressively to search for
#'   nearest neighbors. Default: 4.0.
#' @param a,b More specific parameters controlling the embedding. If not set,
#'   then these values are set automatically as determined by \code{min_dist}
#'   and \code{spread}. Default: NULL.
#' @param target_n_neighbors The number of nearest neighbors to use to construct
#'   the target simplcial set. Default: n_neighbors.
#' @param target_metric The metric for measuring distance between the actual and
#'   and the target values (\code{y}) if using supervised dimension reduction.
#'   Must be one of {"categorical", "euclidean"}. Default: "categorical".
#' @param target_weight Weighting factor between data topology and target
#'   topology. A value of 0.0 weights entirely on data, a value of 1.0 weights
#'   entirely on target. The default of 0.5 balances the weighting equally
#'   between data and target.
#' @param seed Optional seed for pseudo random number generator.  Default: NULL.
#'   Setting a PRNG seed will enable consistency of trained embeddings, allowing
#'   for reproducible results to 3 digits of precision, but at the expense of
#'   potentially slower training and increased memory usage.
#'   If the PRNG seed is not set, then the trained embeddings will not be
#'   deterministic.
#'
#' @return A UMAP model object that can be used as input to the
#'   \code{cuda_ml_transform()} function.
#'   If \code{transform_input} is set to TRUE, then the model object will
#'   contain a "transformed_data" attribute containing the lower dimensional
#'   embedding of the input data.
#'
#' @examples
#' library(cuda.ml)
#'
#' model <- cuda_ml_umap(
#'   x = iris[1:4],
#'   y = iris[[5]],
#'   n_components = 2,
#'   n_epochs = 200,
#'   transform_input = TRUE
#' )
#'
#' set.seed(0L)
#' print(kmeans(model$transformed, iter.max = 100, centers = 3))
#' @export
cuda_ml_umap <- function(x, y = NULL, n_components = 2L, n_neighbors = 15L,
                         n_epochs = 500L, learning_rate = 1.0,
                         init = c("spectral", "random"), min_dist = 0.1,
                         spread = 1.0, set_op_mix_ratio = 1.0,
                         local_connectivity = 1L, repulsion_strength = 1.0,
                         negative_sample_rate = 5L, transform_queue_size = 4.0,
                         a = NULL, b = NULL, target_n_neighbors = n_neighbors,
                         target_metric = c("categorical", "euclidean"),
                         target_weight = 0.5, transform_input = TRUE, seed = NULL,
                         cuML_log_level = c("off", "critical", "error", "warn", "info", "debug", "trace")) {
  init <- umap_match_init_mode(init)
  target_metric <- umap_match_metric_type(target_metric)
  cuML_log_level <- match_cuML_log_level(cuML_log_level)

  model_obj <- .umap_fit(
    x = as.matrix(x),
    y = if (length(y) > 0) as.numeric(y) else numeric(0),
    n_components = as.integer(n_components),
    n_neighbors = as.integer(n_neighbors),
    n_epochs = as.integer(n_epochs),
    learning_rate = as.numeric(learning_rate),
    init = init,
    min_dist = as.numeric(min_dist),
    spread = as.numeric(spread),
    set_op_mix_ratio = as.numeric(set_op_mix_ratio),
    local_connectivity = as.integer(local_connectivity),
    repulsion_strength = as.numeric(repulsion_strength),
    negative_sample_rate = as.integer(negative_sample_rate),
    transform_queue_size = as.numeric(transform_queue_size),
    a = a %||% NaN,
    b = b %||% NaN,
    target_n_neighbors = as.integer(target_n_neighbors),
    target_metric = target_metric,
    target_weight = as.numeric(target_weight),
    random_state = as.integer(seed %||% 0L),
    deterministic = !is.null(seed),
    verbosity = cuML_log_level
  )
  model <- new_umap_model(model_obj)

  if (transform_input) {
    model$transformed_data <- cuda_ml_transform(model, x)
  }

  model
}

cuda_ml_get_state.cuda_ml_umap <- function(model) {
  model_state <- .umap_get_state(model)

  new_model_state(model_state, "cuda_ml_umap_model_state")
}

cuda_ml_set_state.cuda_ml_umap_model_state <- function(model_state) {
  model_obj <- .umap_set_state(model_state)

  new_umap_model(model_obj)
}

#' @export
cuda_ml_transform.cuda_ml_umap <- function(model, x, ...) {
  .umap_transform(model = model, x = as.matrix(x))
}
