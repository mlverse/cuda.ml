kmeans_match_init_method <- function(m = c("kmeans++", "random")) {
  if (is.matrix(m)) {
    if (!is.numeric(m)) {
      stop(
        "Initial value of centroids must be specified using a numeric ",
        "matrix."
      )
    } else {
      2L # ML::kmeans::KMeansParams::Array
    }
  } else {
    m <- match.arg(m)
    switch(m, `kmeans++` = 0L, random = 1L)
  }
}

#' Run the k-means clustering algorithm.
#'
#' Run the k-means clustering algorithm.
#'
#' @template model-with-numeric-input
#' @param k The number of clusters.
#' @param max_iters Maximum number of iterations. Default: 300.
#' @param tol Relative tolerance with regards to inertia to declare convergence.
#'   Default: 0 (i.e., do not use inertia-based stopping criterion).
#' @param init_method Method for initializing the centroids. Valid methods
#'   include "kmeans++", "random", or a matrix of k rows, each row specifying
#'   the initial value of a centroid. Default: "kmeans++".
#' @param seed Seed to the random number generator. Default: 0.
#'
#' @return A list containing the cluster assignments and the centroid of each
#'   cluster. Each centroid will be a column within the `centroids` matrix.
#'
#' @examples
#'
#' library(cuda.ml)
#'
#' if (interactive() && cuda_ml_backend_info()$runtime_installed) {
#'   penguins <- palmerpenguins::penguins[
#'     c(
#'       "bill_length_mm", "bill_depth_mm", "flipper_length_mm",
#'       "body_mass_g", "species"
#'     )
#'   ]
#'   penguins <- penguins[complete.cases(penguins), ]
#'   penguin_predictors <- scale(penguins[names(penguins) != "species"])
#'
#'   kclust <- cuda_ml_kmeans(
#'     penguin_predictors,
#'     k = 3, max_iters = 100
#'   )
#'
#'   print(kclust)
#' }
#' @export
cuda_ml_kmeans <- function(
  x,
  k,
  max_iters = 300,
  tol = 0,
  init_method = c("kmeans++", "random"),
  seed = 0L
) {
  init_method_enum <- kmeans_match_init_method(init_method)
  centroids <- matrix(numeric(0))
  if (is.matrix(init_method)) {
    if (nrow(init_method) != k || ncol(init_method) != ncol(x)) {
      stop(
        "Initial value of centroids must be specified with a matrix of k ",
        "rows and (num features) columns, with each row containing the ",
        "initial value of a centroid."
      )
    } else {
      centroids <- init_method
    }
  }
  .kmeans(
    x = as.matrix(x),
    k = as.integer(k),
    max_iters = as.integer(max_iters),
    tol = as.numeric(tol),
    init_method = init_method_enum,
    centroids = centroids,
    seed = as.integer(seed),
    verbosity = 6L
  )
}
