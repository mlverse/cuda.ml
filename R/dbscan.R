#' Run the DBSCAN clustering algorithm.
#'
#' Run the DBSCAN (Density-based spatial clustering of applications with noise)
#' clustering algorithm.
#'
#' @template model-with-numeric-input
#' @param min_pts,eps A point `p` is a core point if at least `min_pts` are
#'   within distance `eps` from it.
#'
#' @return A list containing the cluster assignments of all data points. A data
#'  point not belonging to any cluster (i.e., "noise") will have NA its cluster
#'  assignment.
#'
#' @examples
#' library(cuml4r)
#' library(magrittr)
#'
#' gen_pts <- function() {
#'   centroids <- list(c(1000, 1000), c(-1000, -1000), c(-1000, 1000))
#'
#'   pts <- centroids %>%
#'     purrr::map(
#'       ~ MASS::mvrnorm(10, mu = .x, Sigma = matrix(c(1, 0, 0, 1), nrow = 2))
#'     )
#'
#'   rlang::exec(rbind, !!!pts)
#' }
#'
#' m <- gen_pts()
#' clusters <- cuml_dbscan(m, min_pts = 5, eps = 3)
#'
#' print(clusters)
#' @export
cuml_dbscan <- function(x, min_pts, eps) {
  res <- .dbscan(
    x = as.matrix(x),
    min_pts = min_pts,
    eps = eps,
    max_bytes_per_batch = 0L
  )
  res$labels[which(res$labels == -1)] <- NA

  res
}
