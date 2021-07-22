#' Run the K means clustering algorithm.
#'
#' Run the K means clustering algorithm.
#'
#' @template model-with-numeric-input
#' @param k The number of clusters.
#' @param max_iters Maximum number of iterations (default: 300).
#'
#' @return A list containing the cluster assignments and the centroid of each
#'   cluster. Each centroid will be a column within the `centroids` matrix.
#'
#' @examples
#'
#' library(cuml4r)
#'
#' kclust <- cuml_kmeans(
#'   iris[, which(names(iris) != "Species")],
#'   k = 3,
#'   max_iters = 100
#' )
#'
#' print(kclust)
#' @export
cuml_kmeans <- function(x, k, max_iters = 300) {
  .kmeans(x = as.matrix(x), k = k, max_iters = max_iters)
}
