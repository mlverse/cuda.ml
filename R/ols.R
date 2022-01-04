ols_match_method <- function(method = c("svd", "eig", "qr")) {
  method <- match.arg(method)

  switch(method,
    "svd" = 0L,
    "eig" = 1L,
    "qr" = 2L
  )
}

#' Train a OLS model.
#'
#' Train an Ordinary Least Square (OLS) model for regression tasks.
#'
#' @template supervised-model-inputs
#' @template supervised-model-output
#' @template ellipsis-unused
#' @template fit-intercept
#' @template normalize-input
#' @param method Must be one of {"svd", "eig", "qr"}.
#'
#'   - "svd": compute SVD decomposition using Jacobi iterations.
#'   - "eig": use an eigendecomposition of the covariance matrix.
#'   - "qr": use the QR decomposition algorithm and solve `Rx = Q^T y`.
#'
#'   If the number of features is larger than the sample size, then the
#'   "svd" algorithm will be force-selected because it is the only
#'    algorithm that can support this type of scenario.
#'
#'   Default: "svd".
#'
#' @return A OLS regressor that can be used with the 'predict' S3 generic to
#'   make predictions on new data points.
#'
#' @examples
#'
#' library(cuda.ml)
#'
#' model <- cuda_ml_ols(formula = mpg ~ ., data = mtcars, method = "qr")
#' predictions <- predict(model, mtcars[names(mtcars) != "mpg"])
#'
#' # predictions will be comparable to those from a `stats::lm` model
#' lm_model <- stats::lm(formula = mpg ~ ., data = mtcars, method = "qr")
#' lm_predictions <- predict(lm_model, mtcars[names(mtcars) != "mpg"])
#'
#' print(
#'   all.equal(
#'     as.numeric(lm_predictions),
#'     predictions$.pred,
#'     tolerance = 1e-3
#'   )
#' )
#' @importFrom ellipsis check_dots_used
#' @export
cuda_ml_ols <- function(x, ...) {
  check_dots_used()
  UseMethod("cuda_ml_ols")
}

#' @rdname cuda_ml_ols
#' @export
cuda_ml_ols.default <- function(x, ...) {
  report_undefined_fn("cuda_ml_ols", x)
}

#' @rdname cuda_ml_ols
#' @export
cuda_ml_ols.data.frame <- function(x, y,
                                   method = c("svd", "eig", "qr"),
                                   fit_intercept = TRUE,
                                   normalize_input = FALSE,
                                   ...) {
  processed <- hardhat::mold(x, y)

  cuda_ml_ols_bridge(
    processed = processed,
    method = method,
    fit_intercept = fit_intercept,
    normalize_input = normalize_input
  )
}

#' @rdname cuda_ml_ols
#' @export
cuda_ml_ols.matrix <- function(x, y,
                               method = c("svd", "eig", "qr"),
                               fit_intercept = TRUE,
                               normalize_input = FALSE,
                               ...) {
  processed <- hardhat::mold(x, y)

  cuda_ml_ols_bridge(
    processed = processed,
    method = method,
    fit_intercept = fit_intercept,
    normalize_input = normalize_input
  )
}

#' @rdname cuda_ml_ols
#' @export
cuda_ml_ols.formula <- function(formula, data,
                                method = c("svd", "eig", "qr"),
                                fit_intercept = TRUE,
                                normalize_input = FALSE,
                                ...) {
  processed <- hardhat::mold(formula, data)

  cuda_ml_ols_bridge(
    processed = processed,
    method = method,
    fit_intercept = fit_intercept,
    normalize_input = normalize_input
  )
}

#' @rdname cuda_ml_ols
#' @export
cuda_ml_ols.recipe <- function(x, data,
                               method = c("svd", "eig", "qr"),
                               fit_intercept = TRUE,
                               normalize_input = FALSE,
                               ...) {
  processed <- hardhat::mold(x, data)

  cuda_ml_ols_bridge(
    processed = processed,
    method = method,
    fit_intercept = fit_intercept,
    normalize_input = normalize_input
  )
}

cuda_ml_ols_bridge <- function(processed,
                               method,
                               fit_intercept,
                               normalize_input) {
  validate_lm_input(processed)

  x <- as.matrix(processed$predictors)
  y <- processed$outcomes[[1]]

  method <- ols_match_method(method)

  model_xptr <- .ols_fit(
    x = x,
    y = y,
    fit_intercept = fit_intercept,
    normalize_input = normalize_input,
    algo = method
  )

  new_linear_model(
    cls = "cuda_ml_ols",
    xptr = model_xptr,
    blueprint = processed$blueprint
  )
}
