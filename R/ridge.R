ridge_validate_alpha <- function(alpha) {
  stopifnot(
    "`alpha` must be one positive finite number" = is.numeric(alpha) &&
      length(alpha) == 1L &&
      is.finite(alpha) &&
      alpha > 0
  )
}

#' Train a linear model using ridge regression.
#'
#' Train a linear model with L2 regularization.
#'
#' @template supervised-model-inputs
#' @template supervised-model-output
#' @template ellipsis-unused
#' @template fit-intercept
#' @param alpha Positive multiplier of the L2 penalty term. Use
#'   \code{cuda_ml_ols()} for an unpenalized linear model. Default: 1.
#'
#' @return A ridge regressor that can be used with the 'predict' S3 generic to
#'   make predictions on new data points.
#'
#' @examples
#'
#' library(cuda.ml)
#'
#' if (interactive() && cuda_ml_backend_info()$runtime_installed) {
#'   model <- cuda_ml_ridge(formula = mpg ~ ., data = mtcars, alpha = 1e-3)
#'   cuda_ml_predictions <- predict(model, mtcars[names(mtcars) != "mpg"])
#'
#'   # predictions will be comparable to those from a `glmnet` model with
#'   # `lambda` set to 2e-3 and `alpha` set to 0
#'   # (in `glmnet`, `lambda` is the weight of the penalty term, and `alpha` is
#'   #  the elastic mixing parameter between L1 and L2 penalties.
#'
#'   if (requireNamespace("glmnet", quietly = TRUE)) {
#'     glmnet_model <- glmnet::glmnet(
#'       x = as.matrix(mtcars[names(mtcars) != "mpg"]), y = mtcars$mpg,
#'       alpha = 0, lambda = 2e-3, nlambda = 1, standardize = FALSE
#'     )
#'
#'     glmnet_predictions <- predict(
#'       glmnet_model, as.matrix(mtcars[names(mtcars) != "mpg"]),
#'       s = 0
#'     )
#'
#'     print(
#'       all.equal(
#'         as.numeric(glmnet_predictions),
#'         cuda_ml_predictions$.pred,
#'         tolerance = 1e-3
#'       )
#'     )
#'   }
#' }
#' @importFrom ellipsis check_dots_used
#' @export
cuda_ml_ridge <- function(x, ...) {
  UseMethod("cuda_ml_ridge")
}

#' @rdname cuda_ml_ridge
#' @export
cuda_ml_ridge.default <- function(x, ...) {
  report_undefined_fn("cuda_ml_ridge", x)
}

#' @rdname cuda_ml_ridge
#' @export
cuda_ml_ridge.data.frame <- function(
  x,
  y,
  alpha = 1,
  fit_intercept = TRUE,
  ...
) {
  check_dots_used()
  processed <- hardhat::mold(x, y)

  cuda_ml_ridge_bridge(
    processed = processed,
    alpha = alpha,
    fit_intercept = fit_intercept
  )
}

#' @rdname cuda_ml_ridge
#' @export
cuda_ml_ridge.matrix <- function(x, y, alpha = 1, fit_intercept = TRUE, ...) {
  check_dots_used()
  processed <- hardhat::mold(x, y)

  cuda_ml_ridge_bridge(
    processed = processed,
    alpha = alpha,
    fit_intercept = fit_intercept
  )
}

#' @rdname cuda_ml_ridge
#' @export
cuda_ml_ridge.formula <- function(
  formula,
  data,
  alpha = 1,
  fit_intercept = TRUE,
  ...
) {
  check_dots_used()
  processed <- hardhat::mold(formula, data)

  cuda_ml_ridge_bridge(
    processed = processed,
    alpha = alpha,
    fit_intercept = fit_intercept
  )
}

#' @rdname cuda_ml_ridge
#' @export
cuda_ml_ridge.recipe <- function(
  x,
  data,
  alpha = 1,
  fit_intercept = TRUE,
  ...
) {
  check_dots_used()
  processed <- hardhat::mold(x, data)

  cuda_ml_ridge_bridge(
    processed = processed,
    alpha = alpha,
    fit_intercept = fit_intercept
  )
}

cuda_ml_ridge_bridge <- function(processed, alpha, fit_intercept) {
  validate_lm_input(processed)
  ridge_validate_alpha(alpha)

  x <- as.matrix(processed$predictors)
  y <- processed$outcomes[[1]]

  model_xptr <- .ridge_fit(
    x = x,
    y = y,
    fit_intercept = fit_intercept,
    alpha = as.numeric(alpha),
    # TODO: future versions of libcuml may support multiple algorithms
    algo = 0L
  )

  new_linear_model(
    cls = "cuda_ml_ridge",
    xptr = model_xptr,
    blueprint = processed$blueprint
  )
}
