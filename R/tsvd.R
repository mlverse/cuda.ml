#' Truncated SVD.
#'
#' Dimensionality reduction using Truncated Singular Value Decomposition.
#'
#' @template model-with-numeric-input
#' @template eigen-decomposition
#' @template transform-input
#' @template cuML-log-level
#' @param n_components Desired dimensionality of output data. Must be strictly
#'   less than \code{ncol(x)} (i.e., the number of features in input data).
#'   Default: 2.
#'
#' @return A TSVD model object with the following attributes:
#'   - "components": a matrix of \code{n_components} rows to be used for
#'      dimensionalitiy reduction on new data points.
#'   - "explained_variance": (only present if "transform_input" is set to TRUE)
#'     amount of variance within the input data explained by each component.
#'   - "explained_variance_ratio": (only present if "transform_input" is set to
#'     TRUE) fraction of variance within the input data explained by each
#'     component.
#'    - "singular_values": The singular values corresponding to each component.
#'     The singular values are equal to the 2-norms of the \code{n_components}
#'     variables in the lower-dimensional space.
#'    - "tsvd_params": opaque pointer to TSVD parameters which will be used for
#'      performing inverse transforms.
#'
#' @examples
#' library(cuda.ml)
#'
#' iris.tsvd <- cuda_ml_tsvd(iris[1:4], n_components = 2)
#' print(iris.tsvd)
#' @export
cuda_ml_tsvd <- function(x,
                         n_components = 2L,
                         eig_algo = c("dq", "jacobi"),
                         tol = 1e-7, n_iters = 15L,
                         transform_input = TRUE,
                         cuML_log_level = c("off", "critical", "error", "warn", "info", "debug", "trace")) {
  eig_algo <- match_eig_algo(eig_algo)
  cuML_log_level <- match_cuML_log_level(cuML_log_level)

  model <- .tsvd_fit_transform(
    x = as.matrix(x),
    n_components = as.integer(n_components),
    algo = eig_algo,
    tol = as.numeric(tol),
    n_iters = as.integer(n_iters),
    transform_input = transform_input,
    verbosity = cuML_log_level
  )
  class(model) <- c("cuda_ml_tsvd", class(model))

  model
}

#' @export
cuda_ml_transform.cuda_ml_tsvd <- function(model, x, ...) {
  .tsvd_transform(model = model, x = as.matrix(x))
}

#' @export
cuda_ml_inverse_transform.cuda_ml_tsvd <- function(model, x, ...) {
  .tsvd_inverse_transform(model = model, x = as.matrix(x))
}
