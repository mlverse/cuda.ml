#' Run the K means clustering algorithm.
#'
#' Run the K means clustering algorithm.
#'
#' @param m The input matrix. Each data point should be a column within this matrix.
#' @param k The number of clusters.
#'
#' @param max_iters Maximum number of iterations (default: 300).
#'
#' @return A list containing the cluster assignments and the centroid of each
#'   cluster. Each centroid will be a column within the `centroids` matrix.
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
#'   rlang::exec(rbind, !!!pts) %>%
#'     t()
#' }
#' 
#' m <- gen_pts()
#' res <- cuml_kmeans(m, k = 3, max_iters = 100)
#' 
#' print(res)
#'
#' @export
cuml_kmeans <- function(m, k, max_iters = 300) {
  .kmeans(m = m, k = k, max_iters = max_iters)
}
