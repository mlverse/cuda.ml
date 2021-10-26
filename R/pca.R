new_pca_model <- function(model) {
  class(model) <- c("cuda_ml_pca", "cuda_ml_model", class(model))

  model
}

#' Perform principal component analysis.
#'
#' Compute principal component(s) of the input data. Each feature from the input
#' will be mean-centered (but not scaled) before the SVD computation takes
#' place.
#'
#' @template model-with-numeric-input
#' @template eigen-decomposition
#' @template transform-input
#' @template cuML-log-level
#' @param n_components Number of principal component(s) to keep. Default:
#'   min(nrow(x), ncol(x)).
#' @param whiten If TRUE, then de-correlate all components, making each
#'   component have unit variance  and removing multi-collinearity.
#'   Default: FALSE.
#'
#' @return A PCA model object with the following attributes:
#'    - "components": a matrix of \code{n_components} rows containing the top
#'      principal components.
#'    - "explained_variance": amount of variance within the input data explained
#'      by each component.
#'    - "explained_variance_ratio": fraction of variance within the input data
#'      explained by each component.
#'    - "singular_values": singular values (non-negative) corresponding to the
#'      top principal components.
#'    - "mean": the column wise mean of \code{x} which was used to mean-center
#'      \code{x} first.
#'    - "transformed_data": (only present if "transform_input" is set to TRUE)
#'      an approximate representation of input data based on principal
#'      components.
#'    - "pca_params": opaque pointer to PCA parameters which will be used for
#'      performing inverse transforms.
#'
#'  The model object can be used as input to the inverse_transform() function to
#'  map a representation based on principal components back to the original
#'  feature space.
#'
#' @examples
#'
#' library(cuda.ml)
#'
#' iris.pca <- cuda_ml_pca(iris[1:4], n_components = 3)
#' print(iris.pca)
#' @export
cuda_ml_pca <- function(x,
                        n_components = NULL,
                        eig_algo = c("dq", "jacobi"),
                        tol = 1e-7, n_iters = 15L,
                        whiten = FALSE,
                        transform_input = TRUE,
                        cuML_log_level = c("off", "critical", "error", "warn", "info", "debug", "trace")) {
  n_components <- n_components %||% min(nrow(x), ncol(x))
  eig_algo <- match_eig_algo(eig_algo)
  cuML_log_level <- match_cuML_log_level(cuML_log_level)

  model_obj <- .pca_fit_transform(
    x = as.matrix(x),
    n_components = as.integer(n_components),
    algo = eig_algo,
    tol = as.numeric(tol),
    n_iters = as.integer(n_iters),
    whiten = whiten,
    transform_input = transform_input,
    verbosity = cuML_log_level
  )

  new_pca_model(model_obj)
}

#' @export
cuda_ml_inverse_transform.cuda_ml_pca <- function(model, x, ...) {
  .pca_inverse_transform(model = model, x = as.matrix(x))
}

cuda_ml_get_state.cuda_ml_pca <- function(model) {
  model_state <- .pca_get_state(model)

  new_model_state(model_state, "cuda_ml_pca_model_state")
}

cuda_ml_set_state.cuda_ml_pca_model_state <- function(model_state) {
  model_state <- .pca_set_state(model_state)

  new_pca_model(model_state)
}
