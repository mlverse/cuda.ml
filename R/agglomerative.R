agglomerative_clustering_match_metric <- function(metric = c("euclidean", "l1", "l2", "manhattan", "cosine")) {
  metric <- match.arg(metric)

  switch(metric,
    euclidean = 1L,
    l1 = 3L,
    l2 = 1L,
    manhattan = 3L,
    cosine = 2L
  )
}

#' Perform Single-Linkage Agglomerative Clustering.
#'
#' Recursively merge the pair of clusters that minimally increases a given
#' linkage distance.
#'
#' @template model-with-numeric-input
#' @param n_clusters The number of clusters to find. Default: 2L.
#' @param metric Metric used for linkage computation. Must be one of
#'   {"euclidean", "l1", "l2", "manhattan", "cosine"}. If connectivity is
#'   "knn" then only "euclidean" is accepted. Default: "euclidean".
#' @param connectivity The type of connectivity matrix to compute. Must be one
#'   of {"pairwise", "knn"}. Default: "pairwise".
#'     - 'pairwise' will compute the entire fully-connected graph of pairwise
#'        distances between each set of points. This is the fastest to compute
#'        and can be very fast for smaller datasets but requires O(n^2) space.
#'     - 'knn' will sparsify the fully-connected connectivity matrix to save
#'       memory and enable much larger inputs. "n_neighbors" will control the
#'       amount of memory used and the graph will be connected automatically in
#'       the event "n_neighbors" was not large enough to connect it.
#' @param n_neighbors The number of neighbors to compute when
#'   \code{connectivity} is "knn". Default: 15L.
#'
#' @return A clustering object with the following attributes:
#'   "n_clusters": The number of clusters found by the algorithm.
#'   "children": The children of each non-leaf node. Values less than
#'     \code{nrow(x)} correspond to leaves of the tree which are the original
#'     samples. \code{children[i + 1][1]} and \code{children[i + 1][2]} were
#'     merged to form node \code{(nrow(x) + i)} in the \code{i}-th iteration.
#'   "labels": cluster label of each data point.
#'
#' @examples
#'
#' library(cuda.ml)
#' library(MASS)
#' library(magrittr)
#' library(purrr)
#'
#' set.seed(0L)
#'
#' gen_pts <- function() {
#'   centers <- list(c(1000, 1000), c(-1000, -1000), c(-1000, 1000))
#'   pts <- centers %>%
#'     map(~ mvrnorm(50, mu = .x, Sigma = diag(2)))
#'
#'   rlang::exec(rbind, !!!pts) %>% as.matrix()
#' }
#'
#' clust <- cuda_ml_agglomerative_clustering(
#'   x = gen_pts(),
#'   metric = "euclidean",
#'   n_clusters = 3L
#' )
#'
#' print(clust$labels)
#' @export
cuda_ml_agglomerative_clustering <- function(x, n_clusters = 2L,
                                             metric = c("euclidean", "l1", "l2", "manhattan", "cosine"),
                                             connectivity = c("pairwise", "knn"),
                                             n_neighbors = 15L) {
  metric <- agglomerative_clustering_match_metric(metric)
  connectivity <- match.arg(connectivity)

  .agglomerative_clustering(
    x = as.matrix(x),
    pairwise_conn = identical(connectivity, "pairwise"),
    metric = metric,
    n_neighbors = as.integer(n_neighbors),
    n_clusters = as.integer(n_clusters)
  )
}
