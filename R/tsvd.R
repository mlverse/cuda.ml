#' Truncated SVD.
#'
#' Dimensionality reduction using Truncated Singular Value Decomposition.
#'
#' @template model-with-numeric-input
#' @template eigen-decomposition
#' @template transform-input
#' @param n_components Desired dimensionality of output data. Must be strictly
#'   less than \code{ncol(x)} (i.e., the number of features in input data).
#'   Default: 2.
#'
#' @return A TSVD model object with the following attributes:
#'   - "components": a matrix of \code{n_components} rows to be used for
#'      dimensionality reduction on new data points.
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
#' if (interactive() && cuda_ml_backend_info()$runtime_installed) {
#'   oils <- modeldata::oils
#'   oil_predictors <- oils |>
#'     subset(select = -class) |>
#'     scale()
#'
#'   oil_tsvd <- cuda_ml_tsvd(oil_predictors, n_components = 2)
#'   print(oil_tsvd)
#' }
#' @export
cuda_ml_tsvd <- function(
  x,
  n_components = 2L,
  eig_algo = c("dq", "jacobi"),
  tol = 1e-7,
  n_iters = 15L,
  transform_input = TRUE
) {
  eig_algo <- match_eig_algo(eig_algo)

  model <- .tsvd_fit_transform(
    x = as.matrix(x),
    n_components = as.integer(n_components),
    algo = eig_algo,
    tol = as.numeric(tol),
    n_iters = as.integer(n_iters),
    transform_input = transform_input,
    verbosity = 0L
  )
  class(model) <- c("cuda_ml_tsvd", class(model))

  model
}

#' @export
cuda_ml_transform.cuda_ml_tsvd <- function(model, x, ...) {
  x <- as.matrix(x)
  stopifnot(
    "`x` must have the same number of columns as the fitted TSVD input" =
      ncol(x) == ncol(model$components)
  )

  .tsvd_transform(model = model, x = x)
}

#' @export
cuda_ml_inverse_transform.cuda_ml_tsvd <- function(model, x, ...) {
  x <- as.matrix(x)
  stopifnot(
    "`x` must have one column per fitted TSVD component" =
      ncol(x) == nrow(model$components)
  )

  .tsvd_inverse_transform(model = model, x = x)
}
