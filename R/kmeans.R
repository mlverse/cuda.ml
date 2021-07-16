#' Run the K means clustering algorithm.
#'
#' Run the K means clustering algorithm.
#'
#' @param m The input matrix or dataframe. Each data point should be a row
#'   and should consist of numeric values only.
#' @param k The number of clusters.
#'
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
#'
#' @export
cuml_kmeans <- function(m, k, max_iters = 300) {
  .kmeans(m = as.matrix(m), k = k, max_iters = max_iters)
}
