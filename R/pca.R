match_eig_algo <- function(eig_algo = c("dq", "jacobi")) {
  eig_algo <- match.arg(eig_algo)

  switch(eig_algo,
    dq = 0L,
    jacobi = 1L
  )
}

#' Perform principal component analysis.
#'
#' Compute principal component(s) of the input data.
#'
#' @template model-with-numeric-input
#' @template cuml-log-level
#' @param n_components Number of principal component(s) to keep. Default:
#'   min(nrow(x), ncol(x)).
#' @param algo Eigen decomposition algorithm to be applied to the covariance
#'   matrix. Valid choices are "dq" (divid-and-conquer method for symmetric
#'   matrices) and "jacobi" (the Jacobi method for symmetric matrices).
#'   Default: "dq".
#' @param tol Tolerance for singular values computed by the Jacobi method.
#'   Default: 1e-7.
#' @param n_iters Maximum number of iterations for the Jacobi method.
#'   Default: 15.
#' @param whitening If TRUE, then de-correlate all components, making each
#'   component have unit variance  and removing multi-collinearity.
#'   Default: FALSE.
#'
#' @return A named list with the following elements:
#'    - "components": a matrix of n_components rows containing the top principal
#'      components.
#'    - "explained_variance": amount of variance within the input data explained
#'      by each component.
#'    - "singular_values": singular values (non-negative) corresponding to the
#'      top principal components.
#'    - "mean": The column wise mean of \code{x} which was used to mean-center
#'      \code{x} first.
#'
#' @examples
#'
#' library(cuml4r)
#'
#' iris.pca <- cuml_pca(iris[1:4], n_components = 3)
#' print(iris.pca)
#' @export
cuml_pca <- function(x,
                     n_components = NULL,
                     eig_algo = c("dq", "jacobi"),
                     tol = 1e-7, n_iters = 15L,
                     whiten = FALSE,
                     cuml_log_level = c("off", "critical", "error", "warn", "info", "debug", "trace")) {
  n_components <- n_components %||% min(nrow(x), ncol(x))
  eig_algo <- match_eig_algo(eig_algo)
  cuml_log_level <- match_cuml_log_level(cuml_log_level)

  .pca_fit_transform(
    x = as.matrix(x),
    n_components = as.integer(n_components),
    algo = eig_algo,
    tol = as.numeric(tol),
    n_iters = n_iters,
    whiten = whiten,
    verbosity = cuml_log_level
  )
}
